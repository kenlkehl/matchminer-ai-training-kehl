#!/usr/bin/env python
"""
Train a ModernBERT-base multi-task classification model on imaging reports.
Supports training, validation, test inference, and eval-only modes.

Usage:
    source ~/gptoss2/bin/activate
    python train_imaging_teacher.py [--args]
"""

import os
import argparse
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_scheduler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from tqdm import tqdm

from teacher_label_utils import (
    cancer_presence_label,
    field_code,
    labels_are_usable,
    merge_labels_with_notes,
    parse_labels_json,
    progression_label,
    read_table,
    response_label,
    split_train_val_test,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

OUTCOME_COLUMNS = [
    'any_cancer', 'response', 'progression',
    'brain_involved_with_cancer',
    'bone_involved_with_cancer',
    'adrenal_involved_with_cancer',
    'liver_involved_with_cancer',
    'lung_involved_with_cancer',
    'node_involved_with_cancer',
    'peritoneal_involved_with_cancer',
]
SITE_COLUMNS = OUTCOME_COLUMNS[3:]
LEGACY_SITE_COLUMN_MAP = {
    'brain_met': 'brain_involved_with_cancer',
    'bone_met': 'bone_involved_with_cancer',
    'adrenal_met': 'adrenal_involved_with_cancer',
    'liver_met': 'liver_involved_with_cancer',
    'lung_met': 'lung_involved_with_cancer',
    'node_met': 'node_involved_with_cancer',
    'peritoneal_met': 'peritoneal_involved_with_cancer',
}
MODEL_NAME = 'answerdotai/ModernBERT-base'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../data/no_phi'))
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, 'synthetic_imaging_vllm_labeled.parquet')
DEFAULT_NOTES_PATH = os.path.join(DATA_DIR, 'synthetic_imaging.parquet')


def parse_args():
    parser = argparse.ArgumentParser(description='Train ModernBERT imaging teacher model')

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=0, help='LR scheduler warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='AdamW weight decay')
    parser.add_argument('--max_seq_length', type=int, default=1500, help='Max tokenizer sequence length')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Classification head intermediate dim')

    # Device
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')

    # Paths
    parser.add_argument('--data_path', type=str,
                        default=DEFAULT_DATA_PATH,
                        help='Path to VLLM-labeled synthetic imaging parquet, or legacy flat CSV/parquet')
    parser.add_argument('--notes_path', type=str,
                        default=DEFAULT_NOTES_PATH,
                        help='Synthetic imaging notes parquet used to join text/split onto VLLM label outputs')
    parser.add_argument('--model_dir', type=str, default=os.path.join(SCRIPT_DIR, '../../../models'),
                        help='Directory to save model weights')
    parser.add_argument('--model_name', type=str, default='modernbert_imaging_teacher.model',
                        help='Model weights filename')
    parser.add_argument('--output_dir', type=str, default=os.path.join(SCRIPT_DIR, '../../../output_data'),
                        help='Directory to save inference outputs')
    parser.add_argument('--output_name', type=str, default='modernbert_imaging_teacher_output.parquet',
                        help='Output filename template')

    # Mode flags
    parser.add_argument('--run_test', action='store_true', help='Also run inference on test split')
    parser.add_argument('--eval_only', action='store_true', help='Skip training, just run inference')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Explicit path to load model weights (defaults to model_dir/model_name)')

    # DataLoader
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader num_workers')

    args = parser.parse_args()

    # Resolve paths
    args.data_path = os.path.abspath(args.data_path)
    args.notes_path = os.path.abspath(args.notes_path)
    args.model_dir = os.path.abspath(args.model_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    return args


class ImagingDataset(Dataset):
    def __init__(self, df, max_seq_length, model_name=MODEL_NAME):
        self.data = df.reset_index(drop=True)
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        encoded = self.tokenizer(
            row['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors=None,
        )

        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long)

        labels = []
        for col in OUTCOME_COLUMNS:
            val = row[col]
            if pd.isna(val):
                labels.append(torch.tensor(float('nan'), dtype=torch.float32))
            else:
                labels.append(torch.tensor(float(val), dtype=torch.float32))

        return (input_ids, attention_mask, *labels)


class ImagingTeacherModel(nn.Module):
    def __init__(self, hidden_dim=128, model_name=MODEL_NAME):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(768, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for name in OUTCOME_COLUMNS
        })

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0, :]
        return [self.heads[name](pooled) for name in OUTCOME_COLUMNS]


def has_synthetic_labels(df):
    return 'labels_json' in df.columns


def text_blob(*parts):
    return ' '.join(str(part) for part in parts if part not in (None, '')).lower()


def site_is_excluded(site):
    blob = text_blob(
        site.get('site_text'),
        site.get('icdo_topography_label'),
        site.get('basis'),
        site.get('evidence'),
    )
    return any(
        re.search(pattern, blob)
        for pattern in (
            r'\bresolved\b',
            r'\bno longer\b',
            r'\bindeterminate\b',
            r'\bequivocal\b',
            r'\bpossible\b',
            r'\bquestionable\b',
            r'\brule out\b',
        )
    )


def site_targets_from_labels(labels, any_cancer):
    values = {name: float('nan') for name in SITE_COLUMNS}
    if pd.isna(any_cancer):
        return values
    values = {name: 0.0 for name in SITE_COLUMNS}
    if any_cancer != 1:
        return values

    sites = labels.get('image_cancer_sites') or []
    if not isinstance(sites, list):
        return values

    for site in sites:
        if not isinstance(site, dict) or site_is_excluded(site):
            continue
        code = str(site.get('icdo_topography_code') or site.get('code') or '').upper().strip()
        blob = text_blob(
            site.get('site_text'),
            site.get('icdo_topography_label'),
            site.get('basis'),
            site.get('evidence'),
        )

        if code.startswith(('C70', 'C71')) or re.search(r'\b(brain|cerebr|cerebell|intracranial|leptomening|meningeal|dural)\b', blob):
            values['brain_involved_with_cancer'] = 1.0
        if code.startswith(('C40', 'C41')) or re.search(r'\b(bone|osseous|skeletal|spine|spinal|vertebr|rib|skull|sacrum|iliac|femur|humerus|sclerotic|lytic)\b', blob):
            values['bone_involved_with_cancer'] = 1.0
        if code.startswith('C74') or re.search(r'\badrenal\b', blob):
            values['adrenal_involved_with_cancer'] = 1.0
        if code.startswith('C22') or re.search(r'\b(liver|hepatic)\b', blob):
            values['liver_involved_with_cancer'] = 1.0
        if code.startswith('C34') or re.search(r'\b(lung|pulmonary)\b', blob):
            values['lung_involved_with_cancer'] = 1.0
        if code.startswith('C77') or re.search(r'\b(lymph|node|nodes|nodal|adenopathy|lymphadenopathy)\b', blob):
            values['node_involved_with_cancer'] = 1.0
        if code.startswith(('C48.1', 'C48.2', 'C48.8')) or re.search(r'\b(peritone|omentum|omental|mesenter|carcinomatosis|ascites)\b', blob):
            values['peritoneal_involved_with_cancer'] = 1.0

    return values


def flatten_synthetic_labels(df, notes_path):
    df = merge_labels_with_notes(df, notes_path)
    rows = []
    for _, row in df.iterrows():
        labels = parse_labels_json(row.get('labels_json'))
        flattened = {name: float('nan') for name in OUTCOME_COLUMNS}
        if labels and labels_are_usable(labels):
            any_cancer = cancer_presence_label(field_code(labels, 'image_ca'))
            status = field_code(labels, 'image_overall')
            flattened['any_cancer'] = any_cancer
            flattened['response'] = response_label(any_cancer, status)
            flattened['progression'] = progression_label(any_cancer, status)
            flattened.update(site_targets_from_labels(labels, any_cancer))
        rows.append(flattened)

    label_df = pd.DataFrame(rows)
    df = pd.concat([df.reset_index(drop=True), label_df], axis=1)
    df['text'] = df['synthetic_note'].fillna('').astype(str).str.lower()
    return df


def prepare_legacy_labels(df):
    for old_name, new_name in LEGACY_SITE_COLUMN_MAP.items():
        if new_name not in df.columns and old_name in df.columns:
            df[new_name] = df[old_name]

    if 'class_status' in df.columns and 'progression' in df.columns:
        df.loc[df['class_status'] == 3, 'progression'] = 1

    if 'text' not in df.columns:
        if 'synthetic_note' in df.columns:
            df['text'] = df['synthetic_note']
        elif 'report_text' in df.columns:
            df['text'] = df['report_text']
        else:
            raise SystemExit("Legacy imaging data must contain text, synthetic_note, or report_text.")

    return df


def load_data(data_path, notes_path=None):
    print(f'Loading data from {data_path}...')
    df = read_table(data_path)
    print(f'  Total rows: {len(df)}')

    if has_synthetic_labels(df):
        if not notes_path:
            raise SystemExit("--notes_path is required when --data_path contains nested VLLM labels.")
        df = flatten_synthetic_labels(df, notes_path)
    else:
        df = prepare_legacy_labels(df)

    missing = [name for name in OUTCOME_COLUMNS if name not in df.columns]
    if missing:
        raise SystemExit(f"Training data is missing outcome column(s): {', '.join(missing)}")

    for name in OUTCOME_COLUMNS:
        df[name] = pd.to_numeric(df[name], errors='coerce')
    df['text'] = df['text'].fillna('').astype(str).str.lower()

    train_df, val_df, test_df = split_train_val_test(df)

    print(f'  Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}')
    for name in OUTCOME_COLUMNS:
        valid = df[name].notna().sum()
        positives = int((df[name] == 1).sum())
        print(f'  {name}: valid={valid}, positive={positives}')
    return train_df, val_df, test_df


def train_model(model, train_df, val_df, args, device):
    train_dataset = ImagingDataset(train_df, args.max_seq_length)
    val_dataset = ImagingDataset(val_df, args.max_seq_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        'linear', optimizer=optimizer,
        num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps,
    )

    model.to(device)

    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        running_losses = {name: 0.0 for name in OUTCOME_COLUMNS}
        running_total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs} [Train]')
        for batch in pbar:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            true_labels = [batch[i + 2].to(device) for i in range(len(OUTCOME_COLUMNS))]

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)

            losses = []
            for i, name in enumerate(OUTCOME_COLUMNS):
                mask = ~torch.isnan(true_labels[i])
                if mask.any():
                    loss = F.binary_cross_entropy_with_logits(
                        preds[i].squeeze(1)[mask], true_labels[i][mask],
                    )
                    losses.append(loss)
                    running_losses[name] += loss.item()

            if losses:
                total_loss = sum(losses)
                total_loss.backward()
                optimizer.step()
                lr_scheduler.step()

                running_total_loss += total_loss.item()
                n_batches += 1
                pbar.set_postfix({'loss': f'{running_total_loss / n_batches:.4f}'})

        mean_train_loss = running_total_loss / max(n_batches, 1)
        print(f'\n  Epoch {epoch + 1} train loss: {mean_train_loss:.4f}')
        for name in OUTCOME_COLUMNS:
            print(f'    {name}: {running_losses[name] / max(n_batches, 1):.4f}')

        # --- Validation ---
        model.eval()
        val_losses = {name: 0.0 for name in OUTCOME_COLUMNS}
        val_total_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{args.epochs} [Val]'):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                true_labels = [batch[i + 2].to(device) for i in range(len(OUTCOME_COLUMNS))]

                preds = model(input_ids, attention_mask)

                batch_loss = 0.0
                for i, name in enumerate(OUTCOME_COLUMNS):
                    mask = ~torch.isnan(true_labels[i])
                    if mask.any():
                        loss = F.binary_cross_entropy_with_logits(
                            preds[i].squeeze(1)[mask], true_labels[i][mask],
                        )
                        val_losses[name] += loss.item()
                        batch_loss += loss.item()

                val_total_loss += batch_loss
                val_batches += 1

        mean_val_loss = val_total_loss / max(val_batches, 1)
        print(f'\n  Epoch {epoch + 1} val loss: {mean_val_loss:.4f}')
        for name in OUTCOME_COLUMNS:
            print(f'    {name}: {val_losses[name] / max(val_batches, 1):.4f}')

    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    save_path = os.path.join(args.model_dir, args.model_name)
    torch.save(model.state_dict(), save_path)
    print(f'\nModel saved to {save_path}')


def run_inference(model, df, args, device):
    dataset = ImagingDataset(df, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model.eval()

    all_probs = {name: [] for name in OUTCOME_COLUMNS}
    all_labels = {name: [] for name in OUTCOME_COLUMNS}

    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference'):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            true_labels = [batch[i + 2] for i in range(len(OUTCOME_COLUMNS))]

            preds = model(input_ids, attention_mask)

            for i, name in enumerate(OUTCOME_COLUMNS):
                probs = torch.sigmoid(preds[i].squeeze(1)).cpu().numpy()
                labels = true_labels[i].numpy()
                all_probs[name].append(probs)
                all_labels[name].append(labels)

    for name in OUTCOME_COLUMNS:
        all_probs[name] = np.concatenate(all_probs[name])
        all_labels[name] = np.concatenate(all_labels[name])

    return all_labels, all_probs


def find_best_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns n+1 precision/recall values; last threshold is implicit
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1)
    return thresholds[best_idx], f1[best_idx]


def compute_metrics(y_true, y_prob, threshold):
    metrics = {}
    metrics['auroc'] = roc_auc_score(y_true, y_prob)
    metrics['auprc'] = average_precision_score(y_true, y_prob)

    y_pred_binary = (y_prob >= threshold).astype(int)

    tp = np.sum((y_pred_binary == 1) & (y_true == 1))
    tn = np.sum((y_pred_binary == 0) & (y_true == 0))
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true == 1))

    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return metrics


def get_output_paths(args, split_name):
    base_name = args.output_name
    if 'output' in base_name:
        parquet_name = base_name.replace('output', f'{split_name}_output', 1)
    else:
        parquet_name = f'{split_name}_{base_name}'

    metrics_name = parquet_name.replace('.parquet', '_metrics.txt')

    parquet_path = os.path.join(args.output_dir, parquet_name)
    metrics_path = os.path.join(args.output_dir, metrics_name)
    return parquet_path, metrics_path


def evaluate_and_save(all_labels, all_probs, original_df, args, split_name, val_thresholds=None):
    """Compute metrics, save output parquet and metrics txt. Returns thresholds dict."""
    os.makedirs(args.output_dir, exist_ok=True)
    parquet_path, metrics_path = get_output_paths(args, split_name)

    thresholds = {}
    all_metrics = {}

    # Build output DataFrame starting from original columns
    out_df = original_df.reset_index(drop=True).copy()

    for name in OUTCOME_COLUMNS:
        y_true = all_labels[name]
        y_prob = all_probs[name]

        # Add continuous predictions
        out_df[f'{name}_pred'] = y_prob

        # Filter out NaN labels for metrics
        valid_mask = ~np.isnan(y_true)
        y_true_valid = y_true[valid_mask]
        y_prob_valid = y_prob[valid_mask]

        if len(y_true_valid) == 0 or len(np.unique(y_true_valid)) < 2:
            print(f'  Warning: {name} has insufficient valid labels for metrics in {split_name}')
            thresholds[name] = 0.5
            all_metrics[name] = None
            out_df[f'{name}_pred_binary'] = (y_prob >= 0.5).astype(int)
            continue

        # Get threshold
        if val_thresholds is not None and name in val_thresholds:
            threshold = val_thresholds[name]
            best_f1 = None  # Will compute below
        else:
            threshold, best_f1 = find_best_f1_threshold(y_true_valid, y_prob_valid)

        thresholds[name] = threshold

        # Compute all metrics
        metrics = compute_metrics(y_true_valid, y_prob_valid, threshold)
        metrics['best_f1_threshold'] = threshold

        # Compute F1 at this threshold for this split
        y_pred_binary = (y_prob_valid >= threshold).astype(int)
        p = metrics['ppv']
        r = metrics['sensitivity']
        metrics['best_f1'] = 2 * p * r / (p + r + 1e-10)

        all_metrics[name] = metrics

        # Add binary predictions to output
        out_df[f'{name}_pred_binary'] = (y_prob >= threshold).astype(int)

    # Save parquet
    out_df.to_parquet(parquet_path, index=False)
    print(f'\n  Saved predictions to {parquet_path}')

    # Write metrics file
    with open(metrics_path, 'w') as f:
        f.write(f'Metrics Report: {split_name}\n')
        f.write(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Data: {args.data_path}\n')
        f.write(f'Model: {args.model_path or os.path.join(args.model_dir, args.model_name)}\n')
        f.write(f'Max Seq Length: {args.max_seq_length}\n')
        f.write(f'{"=" * 60}\n\n')

        for name in OUTCOME_COLUMNS:
            f.write(f'=== {name} ===\n')
            if all_metrics[name] is None:
                f.write('  Insufficient valid labels for metrics\n\n')
                continue
            m = all_metrics[name]
            f.write(f'  AUROC:          {m["auroc"]:.4f}\n')
            f.write(f'  AUPRC:          {m["auprc"]:.4f}\n')
            f.write(f'  Best F1:        {m["best_f1"]:.4f}\n')
            f.write(f'  Best Threshold: {m["best_f1_threshold"]:.4f}\n')
            f.write(f'  Sensitivity:    {m["sensitivity"]:.4f}\n')
            f.write(f'  Specificity:    {m["specificity"]:.4f}\n')
            f.write(f'  PPV:            {m["ppv"]:.4f}\n')
            f.write(f'  NPV:            {m["npv"]:.4f}\n')
            f.write(f'  Accuracy:       {m["accuracy"]:.4f}\n')
            f.write('\n')

    print(f'  Saved metrics to {metrics_path}')

    # Print summary
    print(f'\n  {split_name.upper()} METRICS SUMMARY:')
    for name in OUTCOME_COLUMNS:
        if all_metrics[name] is not None:
            m = all_metrics[name]
            print(f'    {name:20s}  AUROC={m["auroc"]:.4f}  AUPRC={m["auprc"]:.4f}  '
                  f'F1={m["best_f1"]:.4f}  Thresh={m["best_f1_threshold"]:.4f}')

    return thresholds


def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.device}')
    print(f'Using device: {device}')

    # Print config
    print(f'\nConfiguration:')
    print(f'  lr={args.lr}, batch_size={args.batch_size}, epochs={args.epochs}')
    print(f'  warmup_steps={args.warmup_steps}, weight_decay={args.weight_decay}')
    print(f'  max_seq_length={args.max_seq_length}, hidden_dim={args.hidden_dim}')
    print(f'  eval_only={args.eval_only}, run_test={args.run_test}')

    # Load data
    train_df, val_df, test_df = load_data(args.data_path, args.notes_path)

    # Initialize model
    print(f'\nInitializing model ({MODEL_NAME})...')
    model = ImagingTeacherModel(hidden_dim=args.hidden_dim)

    if args.eval_only:
        load_path = args.model_path or os.path.join(args.model_dir, args.model_name)
        print(f'Loading model weights from {load_path}')
        model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
        model.to(device)
    else:
        train_model(model, train_df, val_df, args, device)

    # Validation inference
    print(f'\nRunning inference on validation set...')
    val_labels, val_probs = run_inference(model, val_df, args, device)
    val_thresholds = evaluate_and_save(val_labels, val_probs, val_df, args, 'validation')

    # Test inference
    if args.run_test:
        print(f'\nRunning inference on test set...')
        test_labels, test_probs = run_inference(model, test_df, args, device)
        evaluate_and_save(test_labels, test_probs, test_df, args, 'test', val_thresholds=val_thresholds)

    print('\nDone.')


if __name__ == '__main__':
    main()
