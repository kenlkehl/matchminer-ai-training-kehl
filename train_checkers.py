#!/usr/bin/env python3
"""Train TrialChecker (one ModernBERT with five match-label heads) or BoilerplateChecker.

``trial``: every pair the match teacher labeled in the --labels directories (their scores.jsonl;
each pair once) becomes one example: the inference package's match_quality_checker_template.txt
filled with the cohort's key criteria and the patient's key summary section, and five binary
targets, one per match label (reasonableness, cancer_type, burden, treatment_history, biomarker;
user instruction 2026-11-04). The model is multi-label (one sigmoid head per label, mean BCE) and
names its heads in id2label, so matchminer_ai's score_match_quality returns each head and the
expected match score.

``boilerplate``: every pair the boilerplate teacher labeled (labels.jsonl in --labels) becomes the
inference package's exclusion_criteria_checker_template.txt filled with the patient's boilerplate
section and the cohort's boilerplate criteria, with target excluded (POSITIVE) or not (NEGATIVE):
a separate two-class model, as exclusion_criteria_check expects.

All synthetic pairs train (no holdout; AGENTS.md). Inputs are truncated at --max-length tokens
(default 8,192, ModernBERT's context) and padded per batch. Writes the model, tokenizer and
checker_contract.json (task, labels, template, length, label sources and counts) to --output-model.
Run under torchrun for several GPUs.
"""
import argparse
import hashlib
import json
from pathlib import Path

from trialspace.data import load_corpus, read_jsonl
from trialspace.regression import LABELS

TEMPLATES = {'trial': 'match_quality_checker_template.txt', 'boilerplate': 'exclusion_criteria_checker_template.txt'}
BOILERPLATE_LABELS = ('NEGATIVE', 'POSITIVE')


def template(task):
    from matchminer_ai.llm.prompts import load_prompt_text
    return load_prompt_text(TEMPLATES[task]).strip()  # as the inference checkers load it


def trial_examples(corpus, label_dirs):
    """One example per labeled pair: (text, five binary targets); later rounds win a conflict."""
    patients, cohorts = load_corpus(corpus)
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    fill, pairs, conflicts = template('trial'), {}, 0
    for directory in label_dirs:
        for row in read_jsonl(Path(directory) / 'scores.jsonl'):
            key = (row['patient_id'], row['cohort_id'])
            targets = [1.0 if row['answers'][label] == 'yes' else 0.0 for label in LABELS]
            conflicts += key in pairs and pairs[key] != targets
            pairs[key] = targets
    texts = [fill.format(cmap[cid]['text'], pmap[pid]['text']) for pid, cid in pairs]
    return texts, list(pairs.values()), dict(pairs=len(pairs), conflicting_rounds=conflicts,
                                             positives={label: int(sum(t[i] for t in pairs.values()))
                                                        for i, label in enumerate(LABELS)})


def boilerplate_examples(corpus, label_dirs):
    """One example per labeled pair: (text, 1 when the patient clearly fails a boilerplate criterion)."""
    patients, cohorts = load_corpus(corpus)
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    fill, pairs = template('boilerplate'), {}
    for directory in label_dirs:
        for row in read_jsonl(Path(directory) / 'labels.jsonl'):
            if row.get('task') != 'boilerplate' or type(row.get('excluded')) is not bool:
                raise ValueError('Expected boilerplate labels with a boolean excluded answer')
            pairs[row['patient_id'], row['cohort_id']] = int(row['excluded'])
    texts = [fill.format(pmap[pid]['boilerplate'], cmap[cid]['boilerplate']) for pid, cid in pairs]
    return texts, list(pairs.values()), dict(pairs=len(pairs), excluded=int(sum(pairs.values())))


def train(args):
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    texts, targets, counts = (trial_examples if args.task == 'trial' else boilerplate_examples)(args.corpus, args.labels)
    if not texts:
        raise ValueError('No labeled pairs to train on')
    labels = list(LABELS) if args.task == 'trial' else list(BOILERPLATE_LABELS)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(labels),
        problem_type='multi_label_classification' if args.task == 'trial' else 'single_label_classification',
        id2label=dict(enumerate(labels)), label2id={label: i for i, label in enumerate(labels)})
    model.accepts_loss_kwargs = False  # ModernBERT returns a mean loss and ignores num_items_in_batch

    def tokenize(batch):
        encoded = tokenizer(batch['text'], truncation=True, max_length=args.max_length)
        encoded['labels'] = batch['target']
        return encoded

    dataset = Dataset.from_dict(dict(text=texts, target=targets)).map(tokenize, batched=True, remove_columns=['text', 'target'])
    truncated = sum(len(ids) >= args.max_length for ids in dataset['input_ids'])
    output = Path(args.output_model).expanduser().resolve()
    work = Path(args.work_dir or output.with_name(output.name + '_training')).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(model=model, processing_class=tokenizer, train_dataset=dataset, args=TrainingArguments(
        output_dir=str(work), learning_rate=args.learning_rate, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, warmup_steps=.01, lr_scheduler_type='linear',  # a float is a ratio
        weight_decay=.01, bf16=True, logging_steps=25, save_steps=args.save_steps,
        save_total_limit=2, report_to='none', seed=args.seed, dataloader_num_workers=2))
    trainer.train(resume_from_checkpoint=get_last_checkpoint(str(work)))
    if trainer.is_world_process_zero():
        trainer.save_model(str(output))
        tokenizer.save_pretrained(str(output))
        fill = template(args.task)
        contract = dict(task=args.task, labels=labels, problem_type=model.config.problem_type,
                        template=TEMPLATES[args.task], template_sha256=hashlib.sha256(fill.encode()).hexdigest(),
                        max_length=args.max_length, base_model=args.base_model, corpus=str(Path(args.corpus).resolve()),
                        label_sources=[str(Path(d).resolve()) for d in args.labels], truncated_examples=truncated,
                        epochs=args.epochs, batch_size_per_device=args.batch_size, learning_rate=args.learning_rate,
                        seed=args.seed, holdout='none (all synthetic pairs train)', **counts)
        (output / 'checker_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
        print(f'saved {args.task} checker ({counts["pairs"]} pairs, {truncated} truncated) to {output}', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('task', choices=['trial', 'boilerplate'])
    p.add_argument('--corpus', required=True, help='trialspace_training.py prepare output')
    p.add_argument('--labels', nargs='+', required=True,
                   help='trial: match teacher directories (scores.jsonl); boilerplate: boilerplate teacher directory')
    p.add_argument('--output-model', required=True)
    p.add_argument('--work-dir', help='Checkpoints; default <output-model>_training beside it')
    p.add_argument('--base-model', default='answerdotai/ModernBERT-large')
    p.add_argument('--max-length', type=int, default=8192)
    p.add_argument('--epochs', type=float, default=2)
    p.add_argument('--batch-size', type=int, default=8, help='Pairs per GPU per step')
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--save-steps', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    return p


if __name__ == '__main__':
    train(parser().parse_args())
