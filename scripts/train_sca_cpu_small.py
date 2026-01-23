#!/usr/bin/env python3
"""
SCA Training Script - CPU Optimized (Small Model)
Train a smaller model that works on CPU in reasonable time
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import json

print("🔧 CPU Training - Using Smaller Model")
print("=" * 60)

# Use a smaller model that trains faster on CPU
MODEL_NAME = "distilgpt2"  # 82M parameters (vs 7B)
DATASET_PATH = "../datasets/sca_training_2024_2025.json"

print(f"📥 Loading smaller model: {MODEL_NAME}")
print(f"   Parameters: ~82M (100x smaller than CodeLlama)")
print(f"   Expected CPU training time: 12-24 hours\n")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✅ Model loaded on CPU\n")

# Load dataset
print(f"📂 Loading dataset: {DATASET_PATH}")
dataset = load_dataset('json', data_files=DATASET_PATH)
dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)

print(f"   Training samples: {len(dataset['train'])}")
print(f"   Validation samples: {len(dataset['test'])}\n")

# LoRA configuration (lighter than full fine-tuning)
lora_config = LoraConfig(
    r=8,  # Smaller rank for CPU
    lora_alpha=16,
    target_modules=["c_attn"],  # DistilGPT2 attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize dataset
def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,  # Shorter for CPU
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

print("\n🔄 Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)
print("✅ Dataset tokenized\n")

# Training configuration (CPU optimized)
training_args = TrainingArguments(
    output_dir="./sca-cpu-checkpoints",
    num_train_epochs=1,  # Reduced for CPU
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    fp16=False,  # CPU doesn't support fp16
    save_strategy="epoch",
    logging_steps=50,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    max_grad_norm=1.0,
    report_to="none",
    save_total_limit=1,
    dataloader_num_workers=4,  # Use multiple CPU cores
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

print("🚀 Starting CPU training...")
print("⏰ Estimated time: 12-24 hours")
print("💡 TIP: Run this in screen/tmux session")
print("=" * 60 + "\n")

# Train
trainer.train()

print("\n" + "=" * 60)
print("✅ Training complete!")
print("=" * 60)

# Save model
output_dir = "./sca-cpu-final"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\n💾 Model saved to: {output_dir}")
print("\n⚠️  Note: This smaller model won't be as accurate as CodeLlama-7B")
print("   But it's trainable on CPU in reasonable time!")
