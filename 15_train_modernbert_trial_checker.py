import pandas as pd
import numpy as np
import os
import argparse
import torch
import torch.nn.functional as F
torch.compile.disable = True
torch.set_float32_matmul_precision('high')


def main(checkpoint_dir: str, output_dir: str):

    enrollments = pd.read_parquet('../data/no_phi/space_specific_eligibility_checks.parquet')
    enrollments.info()

    round1_patient = pd.read_parquet('../data/no_phi/round1_patientcentric_checks/top_cohorts_checked_round1.parquet')
    round2_patient = pd.read_parquet('../data/no_phi/round2_patientcentric_checks/top_cohorts_checked_round2.parquet')
    round3_patient = pd.read_parquet('../data/no_phi/round3_patientcentric_checks/top_cohorts_checked_round3.parquet')
    patient = pd.concat([round1_patient, round2_patient, round3_patient], ignore_index=True, axis=0)
    patient.info()

    round1_space = pd.read_parquet('../data/no_phi/round1_trialcentric_checks/top_patients_checked_round1.parquet')
    round2_space = pd.read_parquet('../data/no_phi/round2_trialcentric_checks/top_patients_checked_round2.parquet')
    round3_space = pd.read_parquet('../data/no_phi/round3_trialcentric_checks/top_patients_checked_round3.parquet')

    space = pd.concat([round1_space, round2_space, round3_space], axis=0, ignore_index=True)
    space.info()

    dataset = pd.concat([enrollments, patient, space], axis=0, ignore_index=True).groupby(['patient_summary','this_space']).first().reset_index()

    dataset = dataset[['split', 'patient_summary', 'this_space', 'eligibility_result']]
    dataset = dataset[dataset.eligibility_result >= 0]  # drop parse failures (-1)
    dataset.info()
    print(dataset.eligibility_result.value_counts())

    # Normalize labels to [0, 1] for sigmoid + BCE training
    dataset['label'] = (dataset['eligibility_result'].clip(0, 5).astype('float32')) / 5.0

    from transformers import AutoTokenizer

    dataset.info()

    dataset['pt_trial_pair'] = dataset['this_space'] + "\nNow here is the patient summary:" + dataset['patient_summary']

    dataset=dataset[dataset.split != 'test']
    dataset = dataset[['pt_trial_pair', 'label', 'split']].rename(columns={'pt_trial_pair':'text'})

    from datasets import Dataset, DatasetDict

    train_ds = Dataset.from_pandas(dataset)
    #valid_ds = Dataset.from_pandas(dataset[dataset.split.str.contains('valid')])

    data_dict = DatasetDict({"train":train_ds})

    data_dict

    data_dict['train'][0]

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    #tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=4096)

    tokenized_data = data_dict.map(preprocess_function, batched=True)

    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        #predictions = np.argmax(predictions, axis=1)
        return auroc.compute(predictions=predictions, references=labels)

    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

    class BCETrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            return (loss, outputs) if return_outputs else loss

    model = AutoModelForSequenceClassification.from_pretrained(
        "answerdotai/ModernBERT-large", num_labels=1, reference_compile=False
    )
    model.config.problem_type = "regression"


    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        #evaluation_strategy="epoch",
        save_strategy="epoch",
        #load_best_model_at_end=True,
        push_to_hub=False,
    )


    trainer = BCETrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        #eval_dataset=tokenized_data["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator
        #compute_metrics=compute_metrics,
    )

    # Resume from checkpoint only if checkpoint directory exists
    resume_from_checkpoint = os.path.isdir(checkpoint_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(checkpoint_dir)
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ModernBERT trial checker model")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save training checkpoints")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save final model")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or "../models/trialchecker_regression_checkpoints"
    output_dir = args.output_dir or "../models/modernbert-trial-checker-regression"

    main(checkpoint_dir=checkpoint_dir, output_dir=output_dir)
