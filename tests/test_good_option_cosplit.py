import pandas as pd
import pytest

import train_good_option_checker as core
from train_good_option_cosplit import assign_source_splits
from tests.test_train_good_option_checker import _labels, _catalog


def test_drop_policy_excludes_entire_conflicting_pair():
    result = core.flatten_patient_drug_labels(_labels(conflicting=True), _catalog(), conflict_policy="drop")
    assert result.patient_group_id.tolist() == ["PG2"]
    assert result.attrs["conflict_audit"]["excluded_patient_drug_pairs"] == 1
    assert result.attrs["conflict_audit"]["excluded_assessments"] == 2


def test_modernbert_accumulated_update_matches_full_batch(tmp_path):
    import copy
    import torch
    from datasets import Dataset
    from transformers import ModernBertConfig, ModernBertForSequenceClassification, Trainer, TrainingArguments

    torch.manual_seed(17)
    config = ModernBertConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=32,
        local_attention=8, num_labels=4, problem_type="multi_label_classification",
        pad_token_id=0)
    config._attn_implementation = "sdpa"
    base = ModernBertForSequenceClassification(config)
    data = Dataset.from_dict({"input_ids": [[1,2,3,4], [2,3,4,5], [3,4,5,6], [4,5,6,7]],
        "attention_mask": [[1]*4]*4, "labels": [[1.,0.,1.,0.], [0.,1.,0.,1.], [1.,1.,0.,0.], [0.,0.,1.,1.]]})
    results = []
    for batch_size, accumulation in ((4,1), (2,2)):
        model = copy.deepcopy(base)
        core.configure_mean_bce_accumulation(model)
        args = TrainingArguments(output_dir=str(tmp_path / str(batch_size)), use_cpu=True,
            per_device_train_batch_size=batch_size, gradient_accumulation_steps=accumulation,
            max_steps=1, max_grad_norm=0, report_to="none", save_strategy="no", disable_tqdm=True,
            seed=42, dataloader_pin_memory=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(model=model, args=args, train_dataset=data, optimizers=(optimizer, None))
        trainer.train()
        results.append({name: parameter.detach().clone() for name,parameter in model.named_parameters()})
    for name in results[0]:
        torch.testing.assert_close(results[0][name], results[1][name], atol=1e-6, rtol=1e-5)


def test_original_cosplits_override_mining_train_labels():
    patients = pd.DataFrame({"patient_summary": ["Synthetic A", "Synthetic B", "Synthetic C"],
                             "split": ["train", "Val", "test"]})
    trials = pd.DataFrame({"nct_id": ["T1", "T2", "T3"], "split": ["train", "val", "test"]})
    labels = pd.DataFrame([
        {"patient_group_id": core._hash_text(b"GoodOption patient group", patient),
         "nct_id": trial, "split": "train"}
        for patient in patients.patient_summary for trial in trials.nct_id
    ])
    result = assign_source_splits(labels, patients, trials)
    assert result.partition.tolist() == ["train", "excluded", "excluded", "excluded", "validation",
                                         "excluded", "excluded", "excluded", "excluded"]


def test_conflicting_or_missing_source_splits_fail_closed():
    patients = pd.DataFrame({"patient_summary": ["Synthetic A"], "split": ["train"]})
    labels = pd.DataFrame({"patient_group_id": [core._hash_text(b"GoodOption patient group", "Synthetic A")],
                           "nct_id": ["T1"]})
    with pytest.raises(ValueError, match="Conflicting"):
        assign_source_splits(labels, patients, pd.DataFrame({"nct_id": ["T1", "T1"], "split": ["train", "val"]}))
    with pytest.raises(ValueError, match="no authoritative"):
        assign_source_splits(labels, patients, pd.DataFrame({"nct_id": ["T2"], "split": ["val"]}))


def test_all_labels_prepare_has_no_holdout_or_split_source_reads(tmp_path, monkeypatch):
    import argparse
    import json
    import train_good_option_cosplit as runner

    shards = tmp_path / "shards"
    shards.mkdir()
    _labels().to_parquet(shards / "labels_000000.parquet", index=False)
    monkeypatch.setattr(core, "_inside_no_phi", lambda path: True)
    monkeypatch.setattr(runner, "load_good_option_catalog", lambda path: _catalog())
    args = argparse.Namespace(run_dir=tmp_path / "run", shards=shards,
        catalog=tmp_path / "catalog", patients=tmp_path / "does-not-exist.parquet",
        trials=tmp_path / "does-not-exist.csv", all_labels=True)
    runner.prepare(args)
    data = pd.read_parquet(args.run_dir / "prepared.parquet")
    manifest = json.loads((args.run_dir / "split_manifest.json").read_text())
    assert set(data.partition) == {"train"}
    assert len(data) == 2
    assert manifest["split_strategy"] == "none"
    assert manifest["counts"] == {"train": 2}
    assert manifest["patient_validation_fraction"] == 0
    assert manifest["drug_validation_fraction"] == 0
