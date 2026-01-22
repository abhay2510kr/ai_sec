# Budget Training Guide - Train Without High-End Hardware

## Overview
Complete guide for training CodeLlama 7B models on basic systems or using affordable cloud options. **No expensive GPUs required!**

---

## Table of Contents
1. [Free Cloud GPU Options](#1-free-cloud-gpu-options)
2. [Affordable Cloud GPU Services](#2-affordable-cloud-gpu-services)
3. [Training on CPU/Low-End GPU](#3-training-on-cpulow-end-gpu)
4. [Using Pre-trained Models (Minimal Training)](#4-using-pre-trained-models-minimal-training)
5. [Cost Comparison](#5-cost-comparison)

---

## 1. Free Cloud GPU Options

### 1.1 Google Colab (FREE - Recommended for Beginners)

**Pros:**
- ✅ Completely FREE with T4 GPU (16GB VRAM)
- ✅ No setup required - runs in browser
- ✅ Pre-installed ML libraries
- ✅ Can train smaller batches or single model at a time

**Cons:**
- ⚠️ Session limits (12 hours max runtime)
- ⚠️ May disconnect if idle
- ⚠️ Limited storage (need Google Drive)

**Setup:**

```python
# 1. Go to: https://colab.research.google.com/
# 2. Create new notebook
# 3. Enable GPU: Runtime > Change runtime type > GPU (T4)

# Install dependencies
!pip install torch transformers datasets peft bitsandbytes accelerate axolotl

# Mount Google Drive for datasets
from google.colab import drive
drive.mount('/content/drive')

# Clone your training scripts
!git clone https://github.com/your-repo/ai_sec.git
%cd ai_sec

# Download dataset to Colab
!wget https://your-dataset-url/sast_dataset.parquet

# Run training (example for SAST model)
!python train_sast_colab.py \
  --model_name codellama/CodeLlama-7b-Python-hf \
  --dataset sast_dataset.parquet \
  --output_dir /content/drive/MyDrive/models/sast-model \
  --num_epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 8
```

**Training Script for Colab (Optimized for 16GB VRAM):**

```python
# File: train_sast_colab.py

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# 4-bit quantization config (reduces memory from 28GB to 7GB!)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-Python-hf",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration (only train 0.5% of parameters!)
lora_config = LoraConfig(
    r=16,  # Reduced from 32 for lower memory
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print(f"Trainable parameters: {model.print_trainable_parameters()}")

# Load dataset
dataset = load_dataset("parquet", data_files="sast_dataset.parquet")
dataset = dataset["train"].train_test_split(test_size=0.1)

# Training arguments optimized for Colab
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/models/sast-model",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Small batch for limited memory
    gradient_accumulation_steps=16,  # Effective batch size = 16
    learning_rate=2e-4,
    fp16=True,  # Mixed precision training
    save_strategy="steps",
    save_steps=500,
    logging_steps=10,
    warmup_steps=100,
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
    gradient_checkpointing=True,  # Save memory at cost of speed
    report_to="none",  # Disable W&B if not needed
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Train
trainer.train()

# Save final model
trainer.save_model("/content/drive/MyDrive/models/sast-model-final")
print("Training complete! Model saved to Google Drive.")
```

**Time Estimate on Colab:**
- SAST (80K samples): ~40-50 hours (split into multiple sessions)
- SCA (160K samples): ~60-80 hours
- Smaller models (IaC, Container, API): ~15-25 hours each

**Tips for Colab:**
1. **Save frequently** to Google Drive (Colab can disconnect)
2. **Train one model at a time** across multiple sessions
3. **Use smaller epochs** (3 instead of 5) to fit in 12-hour sessions
4. **Resume training** from checkpoints if disconnected

### 1.2 Kaggle Notebooks (FREE)

**Pros:**
- ✅ FREE 30 hours/week GPU quota (P100 or T4)
- ✅ 20GB RAM + 16GB VRAM
- ✅ Better than Colab for long runs
- ✅ Built-in datasets

**Setup:**

```bash
# 1. Go to: https://www.kaggle.com/
# 2. Create Account > Verify Phone
# 3. Settings > Enable GPU (P100 16GB)
# 4. Create New Notebook

# In Kaggle Notebook:
!pip install peft bitsandbytes accelerate axolotl

# Upload your dataset as Kaggle Dataset
# Add dataset to notebook: Add Data > Your Datasets

# Run training
!python train_sast.py --dataset /kaggle/input/sast-dataset/train.parquet
```

**Time Limit:**
- 9 hours per session
- 30 GPU hours/week (can train ~3-4 models per week)

### 1.3 Lightning AI Studio (FREE Tier)

**Pros:**
- ✅ 22 free GPU hours/month
- ✅ Persistent storage
- ✅ Team collaboration

**Setup:**
```bash
# Go to: https://lightning.ai/
# Create free account
# Launch Studio with GPU
# Same training code as Colab
```

---

## 2. Affordable Cloud GPU Services

### 2.1 Vast.ai (Cheapest Option - $0.20-0.50/hour)

**Best for:** Budget training with flexibility

```bash
# 1. Go to: https://vast.ai/
# 2. Search for GPUs: RTX 3090 (24GB) or RTX 4090

# Example pricing:
# RTX 3090 (24GB): $0.20-0.35/hour
# RTX 4090 (24GB): $0.40-0.60/hour
# A100 (40GB): $0.80-1.20/hour

# 3. Create instance with PyTorch template
# 4. SSH into instance
ssh -p PORT root@IP

# 5. Run training
git clone https://github.com/your-repo/ai_sec.git
cd ai_sec
python train_all_models.py
```

**Cost for ALL 6 models:**
- RTX 3090: ~$20-30 total (100 hours × $0.30/hr)
- RTX 4090: ~$40-60 total
- A100: ~$80-120 total

### 2.2 RunPod (Easy Setup - $0.50-1.00/hour)

**Best for:** Ease of use with reasonable pricing

```bash
# 1. Go to: https://www.runpod.io/
# 2. GPU Pricing:
#    - RTX 4090: $0.69/hour
#    - A6000 (48GB): $0.79/hour
#    - A100 (80GB): $1.89/hour

# 3. Deploy Pod with PyTorch template
# 4. Upload training scripts
# 5. Start training

# Cost for all models: $50-150
```

### 2.3 Lambda Labs (Good GPU Selection - $0.60-1.10/hour)

```bash
# https://lambdalabs.com/service/gpu-cloud
# RTX 6000 Ada: $0.60/hour
# A10 (24GB): $0.60/hour
# A100 (40GB): $1.10/hour

# Cost for all models: $60-110
```

### 2.4 AWS EC2 Spot Instances (70% cheaper than on-demand)

```bash
# g5.xlarge (A10G 24GB): $0.35/hour spot (vs $1.19 on-demand)
# g5.2xlarge: $0.60/hour spot
# p3.2xlarge (V100 16GB): $0.90/hour spot

# Launch spot instance
aws ec2 run-instances \
  --instance-type g5.xlarge \
  --image-id ami-0abcdef1234567890 \
  --spot-price 0.50

# Total cost: $30-90 for all models
```

---

## 3. Training on CPU/Low-End GPU

### 3.1 Training on Your Local Machine (8GB+ RAM)

**Yes, you CAN train on CPU!** It's slow but possible for smaller datasets.

```python
# File: train_cpu_optimized.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load model on CPU
model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-hf",
    torch_dtype=torch.float32,  # Full precision for CPU
    low_cpu_mem_usage=True,
)

# Aggressive LoRA (train even fewer parameters)
lora_config = LoraConfig(
    r=8,  # Very small rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Only 2 modules
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Load SMALL subset of data (10K samples instead of 80K)
dataset = load_dataset("parquet", data_files="sast_dataset.parquet")
small_dataset = dataset["train"].select(range(10000))

# CPU-optimized training args
training_args = TrainingArguments(
    output_dir="./sast-model-cpu",
    num_train_epochs=2,  # Fewer epochs
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    save_steps=1000,
    logging_steps=50,
    optim="adamw_torch",  # CPU-compatible optimizer
    dataloader_num_workers=2,  # Use multiple CPU cores
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_dataset,
)

# This will take DAYS but it WORKS!
trainer.train()
```

**Time on CPU (16-core):**
- Small dataset (10K): ~3-5 days
- Full dataset (80K): ~20-30 days (NOT recommended)

**Recommendation:** Use CPU only for:
- Testing your training pipeline
- Training on very small datasets (5K-10K samples)
- Then move to cloud GPU for full training

### 3.2 Training on Low-End GPU (GTX 1060, RTX 2060, etc.)

If you have a GPU with 6-8GB VRAM:

```python
# Use 4-bit quantization + smallest LoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Minimal LoRA
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj"])

# Tiny batches
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
)
```

**Time on RTX 2060 (8GB):**
- One model: ~5-7 days
- All 6 models: ~30-40 days

---

## 4. Using Pre-trained Models (Minimal Training)

### 4.1 Zero-Shot Prompting (No Training Needed!)

**Option:** Use CodeLlama directly without fine-tuning

```python
# No training required - just use base model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

# Craft good prompts
prompt = """[INST] Analyze the following Python code for SQL injection vulnerabilities:

```python
def get_user(username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    return db.execute(query)
```

Is this code vulnerable? [/INST]"""

# Model responds without any training!
```

**Pros:**
- ✅ Zero cost
- ✅ Immediate use
- ✅ No GPU needed

**Cons:**
- ⚠️ Lower accuracy than fine-tuned models
- ⚠️ May miss domain-specific vulnerabilities

### 4.2 Few-Shot Learning (Minimal Examples)

Add examples in the prompt:

```python
prompt = """[INST] You are a security expert. Here are examples:

Example 1:
Code: eval(user_input)
Vulnerability: Code Injection (CWE-94)

Example 2:
Code: os.system("rm " + filename)
Vulnerability: Command Injection (CWE-78)

Now analyze:
Code: subprocess.call(user_command, shell=True)
Vulnerability: [/INST]"""
```

**Accuracy:** 60-70% (vs 85%+ with fine-tuning)

### 4.3 Lightweight Fine-tuning (100-1000 samples)

Train on just 1000 samples instead of 80,000:

```python
# Select diverse 1000 samples
small_dataset = dataset["train"].shuffle(seed=42).select(range(1000))

# Train for 5 epochs (takes 2-4 hours on Colab)
training_args = TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=4,
)
```

**Accuracy:** 75-80% (vs 85%+ with full dataset)
**Time:** 2-4 hours on free Colab
**Cost:** FREE

---

## 5. Cost Comparison

### Full Training (All 6 Models, 445K samples, ~100 hours)

| Option | Cost | Speed | Setup Difficulty |
|--------|------|-------|------------------|
| **Google Colab (Free)** | $0 | Slow (split sessions) | ⭐ Easy |
| **Kaggle (Free)** | $0 | Slow (30hrs/week) | ⭐ Easy |
| **Vast.ai RTX 3090** | $20-30 | Fast | ⭐⭐ Medium |
| **Vast.ai RTX 4090** | $40-60 | Very Fast | ⭐⭐ Medium |
| **RunPod A100** | $150-200 | Very Fast | ⭐ Easy |
| **Lambda Labs A10** | $60-80 | Fast | ⭐⭐ Medium |
| **AWS Spot g5.xlarge** | $35-50 | Fast | ⭐⭐⭐ Hard |
| **Local CPU (16-core)** | $0 | Very Slow | ⭐ Easy |
| **Local RTX 2060** | $0 | Slow | ⭐ Easy |

### Minimal Training (1K samples per model, ~5-10 hours)

| Option | Cost | Time |
|--------|------|------|
| **Google Colab** | $0 | 8-12 hours |
| **Kaggle** | $0 | 8-12 hours |
| **Vast.ai RTX 3090** | $2-4 | 5-8 hours |
| **Local GPU** | $0 | 12-20 hours |

---

## 6. Recommended Strategy (Best Value)

### For Complete Beginners (FREE):

```
Week 1: Test on Google Colab
- Train 1 model (IaC - smallest dataset)
- Verify pipeline works
- Estimate full training time

Week 2-8: Train on Kaggle
- Use 30 GPU hours/week
- Train one model every 1-2 weeks
- Total: 6-8 weeks, $0 cost
```

### For Best Balance (Under $50):

```
Option A: Vast.ai RTX 4090
- Rent for 100 hours
- Train all 6 models
- Cost: $40-60 total

Option B: RunPod RTX 4090 (weekends only)
- Rent Friday-Sunday (48 hours × 2 weekends)
- Cost: ~$70
- More convenient
```

### For Fastest Results (Under $200):

```
RunPod or Lambda Labs A100
- 80GB VRAM
- Train all models in 3-4 days
- Cost: $150-200
- Best if you need it fast
```

---

## 7. Step-by-Step Beginner Guide

### Option 1: Completely FREE (Google Colab)

**Total Time:** 6-8 weeks (part-time)
**Total Cost:** $0

```
Day 1: Setup
1. Go to https://colab.research.google.com/
2. Create new notebook
3. Runtime > Change runtime type > T4 GPU
4. Copy training script from above
5. Test with 100 samples

Week 1-2: SAST Model
1. Upload dataset to Google Drive
2. Run training (split into 4-5 sessions)
3. Save checkpoints to Drive

Week 3-4: SCA Model
(Repeat process)

Week 5: IaC, Container Models
Week 6: API, DAST Models
```

### Option 2: Fast & Cheap ($30-40)

**Total Time:** 4-5 days
**Total Cost:** $30-40

```
Day 1: Setup Vast.ai
1. Create account: https://vast.ai/
2. Add $40 credit
3. Search: RTX 3090, 24GB VRAM, $0.30/hour
4. Launch instance with PyTorch template
5. SSH into machine

Day 2-5: Training
1. Upload datasets
2. Run training script for all 6 models
3. Download trained models
4. Total: ~100 hours × $0.30 = $30
```

---

## 8. Sample Training Command (All Methods)

```bash
# Works on: Colab, Kaggle, Vast.ai, RunPod, Local

# Install dependencies
pip install torch transformers peft bitsandbytes accelerate

# Train SAST model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    'codellama/CodeLlama-7b-Python-hf',
    quantization_config=bnb_config,
    device_map='auto'
)

tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Python-hf')

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj','v_proj'], lora_dropout=0.05, task_type='CAUSAL_LM')

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

dataset = load_dataset('your_dataset')

training_args = TrainingArguments(
    output_dir='./sast-model',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    fp16=True,
    save_steps=500,
    optim='paged_adamw_8bit',
    gradient_checkpointing=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset['train'])
trainer.train()
trainer.save_model('./sast-model-final')
"
```

---

## Summary

✅ **FREE Options:** Google Colab, Kaggle (0-8 weeks)
✅ **Cheapest Paid:** Vast.ai RTX 3090 ($20-30 total)
✅ **Best Balance:** Vast.ai RTX 4090 ($40-60 total)
✅ **Fastest:** RunPod A100 ($150-200, 3-4 days)
✅ **Local Training:** Possible on CPU/low-end GPU (slower)
✅ **Zero Training:** Use base CodeLlama with good prompts

**My Recommendation:** Start with Google Colab for FREE, train one model to test, then use Vast.ai RTX 4090 for ~$50 to train all 6 models in less than a week. 🚀
