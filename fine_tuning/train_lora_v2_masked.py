import os
# fine_tune_identity.py
# QLoRA fine-tune for Mistral-7B on ~8GB GPU, masked so only assistant tokens count

from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# --- Paths ---
BASE_MODEL_DIR = os.getenv("BASE_MODEL", "models/Mistral-7B")
DATA_JSONL     = os.getenv("TRAIN_DATA", "data/grounded_qa.jsonl")
OUT_ADAPTER    = os.getenv("OUTPUT_DIR", "outputs/lora_v2")
OUT_MERGED     = os.getenv("MERGED_DIR", "outputs/lora_v2_merged")

Path(OUT_ADAPTER).mkdir(parents=True, exist_ok=True)
Path(OUT_MERGED).mkdir(parents=True, exist_ok=True)

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Quantized model load (QLoRA) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id

# --- Prepare for LoRA ---
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# --- Dataset ---
raw = load_dataset("json", data_files=DATA_JSONL)["train"]

MAX_LEN = 512

def format_dataset(ex):
    text = ex["text"].strip()
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LEN)
    labels = enc["input_ids"].copy()

    # Mask out everything up to and including [/INST]
    inst_end = text.find('[/INST]')
    if inst_end != -1:
        inst_tokens = tokenizer(
            text[:inst_end + len("[/INST]")],
            truncation=True,
            max_length=MAX_LEN
        )["input_ids"]
        cutoff = len(inst_tokens)
        labels[:cutoff] = [-100] * cutoff

    # Mask padding too
    for i, m in enumerate(enc["attention_mask"]):
        if m == 0:
            labels[i] = -100

    enc["labels"] = labels
    return enc

train_ds = raw.map(format_dataset, remove_columns=["text"])
print("Train size:", len(train_ds))

# --- Training ---
args = TrainingArguments(
    output_dir=OUT_ADAPTER,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    optim="paged_adamw_8bit",
    remove_unused_columns=False,
    report_to=[],
    dataloader_num_workers=0
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds
)

trainer.train()

# --- Save LoRA adapters ---
trainer.save_model(OUT_ADAPTER)
tokenizer.save_pretrained(OUT_ADAPTER)

# --- Merge LoRA into full model ---
merged = model.merge_and_unload()
merged.save_pretrained(OUT_MERGED, safe_serialization=True)
tokenizer.save_pretrained(OUT_MERGED)

print("Done.\nAdapters:", OUT_ADAPTER, "\nMerged:", OUT_MERGED)
