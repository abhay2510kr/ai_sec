#!/usr/bin/env python3
"""
Vertex AI training script for SCA Package Vulnerability Detection Model
Optimized for Google Cloud Vertex AI with T4 GPU
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from google.cloud import storage

# Environment variables from Vertex AI
MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "/tmp/model")
TENSORBOARD_LOG_DIR = os.environ.get("AIP_TENSORBOARD_LOG_DIR", "/tmp/logs")
CHECKPOINT_DIR = os.environ.get("AIP_CHECKPOINT_DIR", "/tmp/checkpoints")

# Training configuration
DATASET_URL = "https://raw.githubusercontent.com/abhay2510kr/ai_sec/refs/heads/main/datasets/sca_training_2024_2025.json"
MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
MAX_LENGTH = 2048
BATCH_SIZE = 4  # T4 GPU can handle this
GRADIENT_ACCUMULATION = 4
EPOCHS = 3
LEARNING_RATE = 2e-4

def main():
    print("=" * 80)
    print("🚀 Vertex AI Training Job - SCA Vulnerability Detection")
    print("=" * 80)
    
    print(f"\n📊 Configuration:")
    print(f"  - TensorBoard logs: {TENSORBOARD_LOG_DIR}")
    print(f"  - Model output: {MODEL_DIR}")
    print(f"  - Checkpoints: {CHECKPOINT_DIR}")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n🖥️  Hardware:")
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  - CUDA Version: {torch.version.cuda}")
    else:
        print("\n⚠️  WARNING: No GPU detected!")
        return
    
    # Load dataset
    print("\n📥 Loading dataset from GitHub...")
    try:
        dataset = load_dataset('json', data_files={'train': DATASET_URL})
        train_dataset = dataset['train']
        print(f"✅ Loaded {len(train_dataset)} training examples")
        
        # Show sample
        print("\n📝 Sample training example:")
        print(train_dataset[0]['text'][:200] + "...")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Load tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✅ Tokenizer loaded")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )
    
    print("\n🔄 Tokenizing dataset...")
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing"
    )
    print(f"✅ Tokenized {len(tokenized_dataset)} examples")
    
    # Load model with quantization
    print("\n🤖 Loading CodeLlama 7B with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    print("✅ Model loaded and prepared for training")
    
    # Configure LoRA
    print("\n⚙️  Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"✅ LoRA configured:")
    print(f"  - Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  - All params: {all_params:,} ({all_params/1e6:.2f}M)")
    print(f"  - Trainable %: {100 * trainable_params / all_params:.2f}%")
    
    # Training arguments
    print("\n🏋️  Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        fp16=True,
        save_strategy="epoch",
        save_steps=500,
        logging_dir=TENSORBOARD_LOG_DIR,
        logging_steps=10,
        logging_first_step=True,
        report_to="tensorboard",
        save_total_limit=2,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        weight_decay=0.001,
        group_by_length=True,
    )
    
    print(f"✅ Training configuration:")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"  - Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Total steps: {len(tokenized_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION) * EPOCHS}")
    
    # Create trainer
    print("\n👨‍🏫 Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    print("✅ Trainer created")
    
    # Train
    print("\n" + "=" * 80)
    print("🏋️  STARTING TRAINING")
    print("=" * 80)
    
    try:
        trainer.train()
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    
    # Save final model
    print(f"\n💾 Saving model to {MODEL_DIR}...")
    try:
        trainer.save_model(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        print("✅ Model and tokenizer saved")
        
        # Save training info
        with open(os.path.join(MODEL_DIR, "training_info.txt"), "w") as f:
            f.write(f"Model: {MODEL_NAME}\n")
            f.write(f"Dataset: {DATASET_URL}\n")
            f.write(f"Training samples: {len(train_dataset)}\n")
            f.write(f"Epochs: {EPOCHS}\n")
            f.write(f"Learning rate: {LEARNING_RATE}\n")
            f.write(f"Batch size: {BATCH_SIZE}\n")
            f.write(f"Gradient accumulation: {GRADIENT_ACCUMULATION}\n")
            f.write(f"Trainable params: {trainable_params:,}\n")
        print("✅ Training info saved")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        raise
    
    print("\n" + "=" * 80)
    print("🎉 ALL DONE! Your model is ready for deployment!")
    print("=" * 80)
    print(f"\n📂 Model location: {MODEL_DIR}")
    print(f"📊 TensorBoard logs: {TENSORBOARD_LOG_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Download model: gsutil -m cp -r {MODEL_DIR} ./trained_model/")
    print(f"  2. Test inference locally")
    print(f"  3. Deploy to production")
    print("=" * 80)

if __name__ == "__main__":
    main()
