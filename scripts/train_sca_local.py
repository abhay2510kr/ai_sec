#!/usr/bin/env python3
"""
Local training script for SCA Package Vulnerability Detection Model
Trains CodeLlama 7B using 2024-2025 CVE data from GitHub
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

# Configuration
DATASET_URL = "https://raw.githubusercontent.com/abhay2510kr/ai_sec/refs/heads/main/datasets/sca_training_2024_2025.json"
MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
OUTPUT_DIR = "/workspaces/ai_sec/models/sca-package"
MAX_LENGTH = 2048
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
EPOCHS = 3
LEARNING_RATE = 2e-4

def main():
    print("=" * 60)
    print("🔒 SCA Package Vulnerability Model Training")
    print("=" * 60)
    
    # Check GPU/CPU
    print("\n🖥️  Hardware Check:")
    if torch.cuda.is_available():
        print(f"  ✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        use_gpu = True
    else:
        print("  ⚠️  No GPU detected - using CPU (this will be VERY slow!)")
        print("  💡 Recommendation: Use Google Colab for GPU training")
        response = input("\n  Continue with CPU training? (y/n): ")
        if response.lower() != 'y':
            print("  ❌ Training cancelled")
            return
        use_gpu = False
    
    # Download dataset
    print("\n📥 Downloading dataset from GitHub...")
    print(f"  URL: {DATASET_URL}")
    
    try:
        dataset = load_dataset('json', data_files=DATASET_URL, split='train')
        print(f"  ✅ Dataset loaded: {len(dataset)} examples")
    except Exception as e:
        print(f"  ❌ Failed to download dataset: {e}")
        print(f"\n  💡 Make sure the file exists at:")
        print(f"     {DATASET_URL}")
        return
    
    # Split dataset
    print("\n📊 Splitting dataset...")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  Training samples: {len(dataset['train'])}")
    print(f"  Validation samples: {len(dataset['test'])}")
    
    # Show sample
    print(f"\n📝 Sample training example:")
    print(dataset['train'][0]['text'][:400] + "...\n")
    
    # Load model
    print("📥 Loading model (this may take several minutes)...")
    
    if use_gpu:
        # 4-bit quantization for GPU
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("  ✅ Model loaded with 4-bit quantization (~7 GB)")
    else:
        # CPU mode - use smaller precision
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        print("  ✅ Model loaded (CPU mode)")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # LoRA configuration
    print("\n🔧 Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    if use_gpu:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenize dataset
    print("\n🔄 Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    print("  ✅ Tokenization complete")
    
    # Training arguments
    print("\n⚙️  Configuring training...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        fp16=use_gpu,  # Only use fp16 on GPU
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        warmup_steps=50,
        optim="paged_adamw_8bit" if use_gpu else "adamw_torch",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        report_to="none",
        eval_strategy="steps",
        eval_steps=100,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )
    
    # Estimate training time
    steps_per_epoch = len(dataset['train']) // (BATCH_SIZE * GRADIENT_ACCUMULATION)
    total_steps = steps_per_epoch * EPOCHS
    
    if use_gpu:
        est_time = "4-6 hours on T4 GPU"
    else:
        est_time = "24-48+ hours on CPU (NOT RECOMMENDED)"
    
    print(f"\n📈 Training Configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Estimated time: {est_time}")
    print(f"  Output: {OUTPUT_DIR}")
    
    # Start training
    print("\n" + "=" * 60)
    print("🚀 STARTING TRAINING...")
    print("=" * 60)
    print("💡 You can monitor progress in the terminal")
    print("💡 Checkpoints saved every 100 steps")
    print("=" * 60 + "\n")
    
    try:
        trainer.train()
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        
        # Save final model
        final_dir = f"{OUTPUT_DIR}-final"
        print(f"\n💾 Saving final model to: {final_dir}")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        
        print(f"\n✅ Model saved successfully!")
        print(f"\n📁 Model location: {final_dir}")
        print(f"📦 You can now use this model for inference!")
        
        # Test the model
        print("\n🧪 Testing the model...")
        test_input = """<s>[INST] Analyze this npm dependency manifest for known vulnerabilities

```json
{
  "name": "test-app",
  "dependencies": {
    "express": "4.16.0",
    "lodash": "4.17.4"
  }
}
``` [/INST]"""
        
        inputs = tokenizer(test_input, return_tensors="pt")
        if use_gpu:
            inputs = inputs.to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            top_p=0.95
        )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n" + "=" * 60)
        print("🤖 MODEL OUTPUT:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print(f"💾 Checkpoints saved in: {OUTPUT_DIR}")
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
