import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

dataset = Dataset.load_from_disk('../../data/no_phi/oncoreasoning_training_data/tokenized_training_data.dataset/')


lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    use_rslora=True,
    init_lora_weights="gaussian",
    modules_to_save=["lm_head"],
)




repo_id = "google/gemma-4-e2b-it"

torch.backends.cuda.enable_flash_sdp(True)
print(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")

model = AutoModelForCausalLM.from_pretrained(
   repo_id, attn_implementation="sdpa", dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(repo_id)
tokenizer.pad_token = tokenizer.eos_token

sft_config = SFTConfig(
    ## GROUP 1: Memory usage
    # These arguments will squeeze the most out of your GPU's RAM
    # Checkpointing
    gradient_checkpointing=True,    # this saves a LOT of memory when set true but is slower
    # Set this to avoid exceptions in newer versions of PyTorch
    gradient_checkpointing_kwargs={'use_reentrant': False}, 
    # Gradient Accumulation / Batch size
    # Actual batch (for updating) is same (1x) as micro-batch size
    gradient_accumulation_steps=1,  
    # The initial (micro) batch size to start off with
    per_device_train_batch_size=1, 
    bf16=True,
    # If batch size would cause OOM, halves its size until it works
    auto_find_batch_size=False,
    save_total_limit=2,
    #save_safetensors=False,

    ## GROUP 2: Dataset-related
    max_length=tokenizer.max_len_single_sentence,
    # Dataset
    # packing a dataset means no padding is needed
    packing=False,

    ## GROUP 3: These are typical training parameters
    num_train_epochs=1,
    learning_rate=5e-6,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio = 0.10,
    #save_strategy='epoch',
    save_steps=2000,
    #evaluation_strategy='no',
    # Optimizer
    # 8-bit Adam optimizer - doesn't help much if you're using LoRA!
    optim='adamw_torch_fused',       
    dataset_kwargs = {'skip_prepare_dataset':True},
    lr_scheduler_kwargs={'num_cycles':3},
    ## GROUP 4: Logging parameters
    logging_steps=20,
    activation_offloading=True,
    use_liger_kernel=True,
    logging_dir='./logs',
    output_dir='../../models/onco_reasoning_gemma',
    report_to='none'
)

data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    #peft_config = lora_config,
    train_dataset=dataset,
    data_collator=data_collator,
)

len(dataset)





if any(
    d.startswith("checkpoint-") for d in os.listdir(sft_config.output_dir)
) if os.path.isdir(sft_config.output_dir) else False:
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
