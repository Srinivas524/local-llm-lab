import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import Trainer

# === PATHS ===
BASE_MODEL = os.getenv("BASE_MODEL", "models/Mistral-7B")   # local Mistral checkpoint
DATA_FILE = os.getenv("TRAIN_DATA", "data/grounded_qa.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/lora_v1")

# === LOAD DATASET ===
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_FILE)

# === FORMAT DATASET (convert Alpaca-style to single text field) ===
def format_example(example):
    if example.get("input") and len(example["input"].strip()) > 0:
        return {
            "text": f"User: {example['instruction']}\nInput: {example['input']}\nAssistant: {example['output']}"
        }
    else:
        return {
            "text": f"User: {example['instruction']}\nAssistant: {example['output']}"
        }

dataset = dataset.map(format_example)

# === LOAD TOKENIZER & MODEL ===
print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# QLoRA 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype="auto"
)

# ✅ Prepare model for LoRA fine-tuning
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# === LORA CONFIG ===
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Mistral attention
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === TOKENIZE DATASET ===
def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# === TRAINING ARGS ===
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True
)

# === DATA COLLATOR ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# === TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# === TRAIN ===
print("Starting training...")
trainer.train()

# === SAVE ADAPTER SAFETENSORS ===
print("Saving LoRA identity adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Identity adapter training complete! Adapter saved to: {OUTPUT_DIR}")
