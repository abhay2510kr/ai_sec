# Google Cloud Platform - GPU Training Setup Guide

Complete step-by-step guide to train your SCA model on Google Cloud Platform (GCP) with GPU.

---

## Prerequisites

- Google Cloud account ([Create one here](https://cloud.google.com))
- Credit card (for verification - new users get $300 free credits)
- Your GitHub repository with the training notebook

---

## Part 1: Initial GCP Setup (One-time)

### Step 1: Create Google Cloud Account & Project

1. **Go to Google Cloud Console**: https://console.cloud.google.com
2. **Sign up** if you don't have an account (get $300 free credits for 90 days)
3. **Create a new project**:
   - Click "Select a project" → "New Project"
   - Project name: `ai-sec-training`
   - Click "Create"

### Step 2: Enable Required APIs

```bash
# Open Cloud Shell (click >_ icon in top-right corner)
# Run these commands:

# Enable Compute Engine API
gcloud services enable compute.googleapis.com

# Enable Cloud Storage API (for storing models)
gcloud services enable storage-api.googleapis.com
```

### Step 3: Request GPU Quota Increase

**Important**: By default, GPU quota is 0. You must request an increase.

1. Go to: **IAM & Admin** → **Quotas** → **All Quotas**
2. Filter by:
   - Service: "Compute Engine API"
   - Metric: "GPUs (all regions)"
3. Click on the quota → **Edit Quotas**
4. Set **New Limit**: `1` (for 1 GPU)
5. Add request description: "Training ML model for security vulnerability detection"
6. Submit request

**⏰ Wait time**: Usually approved within 5-30 minutes. Check your email.

---

## Part 2: Launch GPU VM Instance

### Option A: Using Cloud Console (Beginner-friendly)

1. **Go to Compute Engine** → **VM Instances** → **Create Instance**

2. **Configure the VM**:

   **Name**: `sca-training-vm`
   
   **Region**: `us-central1` (Iowa - cheapest)
   
   **Zone**: `us-central1-a`
   
   **Machine Configuration**:
   - Series: `N1`
   - Machine type: `n1-standard-4` (4 vCPUs, 15 GB memory)
   
   **GPU**:
   - Click "Add GPU"
   - GPU type: `NVIDIA Tesla T4`
   - Number of GPUs: `1`
   
   **Boot Disk**:
   - Click "Change"
   - Operating System: `Deep Learning on Linux`
   - Version: `Debian 11 based Deep Learning VM with CUDA 11.8 M120`
   - Boot disk type: `SSD persistent disk`
   - Size: `100 GB`
   - Click "Select"
   
   **Firewall**:
   - ✅ Allow HTTP traffic
   - ✅ Allow HTTPS traffic

3. **Click "CREATE"**

**💰 Cost**: ~$0.54/hour (~$3 for 6 hours of training)

### Option B: Using gcloud CLI (Advanced)

```bash
# Open Cloud Shell and run:

gcloud compute instances create sca-training-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=common-cu118-debian-11-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True" \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

---

## Part 3: Connect to Your VM

### Step 1: SSH into the VM

**Method 1: Browser SSH (Easiest)**
1. Go to **Compute Engine** → **VM Instances**
2. Find your VM `sca-training-vm`
3. Click **SSH** button
4. A terminal window will open in your browser

**Method 2: gcloud CLI**
```bash
gcloud compute ssh sca-training-vm --zone=us-central1-a
```

### Step 2: Verify GPU is Available

```bash
# Check NVIDIA driver
nvidia-smi

# Expected output:
# +---------------------------------------------------------------------------------------+
# | NVIDIA-SMI 535.xxx       Driver Version: 535.xxx       CUDA Version: 12.x           |
# |-----------------------------------------+----------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
# |                                         |                      |               MIG M. |
# |=========================================+======================+======================|
# |   0  Tesla T4                       Off | 00000000:00:04.0 Off |                    0 |
# | N/A   35C    P0              25W /  70W |      0MiB / 15360MiB |      0%      Default |
# ...
```

If you see the T4 GPU listed, you're good! ✅

### Step 3: Verify Python & PyTorch

```bash
# Check Python version
python3 --version
# Should be Python 3.10+

# Check if PyTorch can see the GPU
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
# Expected: CUDA Available: True, GPU: Tesla T4
```

---

## Part 4: Setup Training Environment

### Step 1: Clone Your Repository

```bash
# Install git (if not already installed)
sudo apt-get update
sudo apt-get install -y git

# Clone your repository
git clone https://github.com/abhay2510kr/ai_sec.git
cd ai_sec
```

### Step 2: Install Dependencies

```bash
# Install required packages
pip install transformers==4.37.0 datasets==2.16.0 peft==0.8.0 accelerate==0.26.0 sentencepiece torch
```

### Step 3: Download Dataset (Already in Repo)

```bash
# Verify dataset exists
ls -lh datasets/sca_training_2024_2025.json

# Should show: 15M file
```

---

## Part 5: Run Training

### Option A: Convert Colab Notebook to Python Script (Recommended)

```bash
# Create training script
cat > train_sca_gcp.py << 'EOF'
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

print("="*60)
print("🚀 SCA Model Training on GCP GPU")
print("="*60)

# Step 1: Check GPU
print("\n📊 Checking GPU...")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Step 2: Load Dataset
print("\n📂 Loading dataset...")
dataset = load_dataset('json', data_files='datasets/sca_training_2024_2025.json')
dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
print(f"Training samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['test'])}")

# Step 3: Load Model & Tokenizer
print("\n📥 Loading CodeLlama-7b-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-Instruct-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("✅ Model loaded in float16 (~14 GB)")

# Step 4: Configure LoRA
print("\n⚙️  Configuring LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Step 5: Tokenize Dataset
print("\n🔄 Tokenizing dataset...")
def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)
print("✅ Dataset tokenized!")

# Step 6: Configure Training
print("\n⚙️  Configuring trainer...")
training_args = TrainingArguments(
    output_dir="./sca-package-checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="steps",
    save_steps=100,
    logging_steps=10,
    warmup_steps=50,
    optim="adamw_torch",
    max_grad_norm=0.3,
    report_to="tensorboard",  # Enable TensorBoard logging
    evaluation_strategy="steps",
    eval_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Step 7: Start Training
print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("⏱️  Estimated time: 4-6 hours on T4 GPU")
print("💾 Checkpoints saved every 100 steps to: ./sca-package-checkpoints")
print("="*60 + "\n")

trainer.train()

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)

# Step 8: Save Final Model
print("\n💾 Saving final model...")
output_dir = "./sca-package-final"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Model saved to: {output_dir}")

# Step 9: Test Model
print("\n🧪 Testing model...")
test_input = """[INST] Analyze this package.json for known vulnerabilities

```json
{
  "name": "my-app",
  "dependencies": {
    "express": "4.16.0",
    "lodash": "4.17.4",
    "axios": "0.18.0"
  }
}
``` [/INST]"""

inputs = tokenizer(test_input, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True, top_p=0.95)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "="*60)
print("🤖 MODEL OUTPUT:")
print("="*60)
print(result)
print("="*60)

print("\n✅ All done! Model ready for deployment.")
EOF

# Make it executable
chmod +x train_sca_gcp.py
```

### Run Training in Background (Recommended)

```bash
# Start training in tmux session (survives SSH disconnection)
tmux new -s training

# Inside tmux, run:
python3 train_sca_gcp.py 2>&1 | tee training.log

# To detach from tmux: Press Ctrl+B, then D
# To reattach later: tmux attach -t training
# To check progress: tail -f training.log
```

### Option B: Run Jupyter Notebook

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter on port 8888
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# You'll see output like:
# http://127.0.0.1:8888/?token=abc123...

# Create SSH tunnel from your local machine:
# Open NEW terminal on your LOCAL computer and run:
gcloud compute ssh sca-training-vm --zone=us-central1-a -- -L 8888:localhost:8888

# Then open in browser: http://localhost:8888/?token=abc123...
# Navigate to notebooks/train_sca_package_colab.ipynb and run cells
```

---

## Part 6: Monitor Training Progress

### Method 1: TensorBoard (Real-time Charts)

```bash
# In a new tmux window (Ctrl+B, then C):
tensorboard --logdir ./sca-package-checkpoints --host 0.0.0.0 --port 6006

# Create SSH tunnel on your LOCAL machine:
gcloud compute ssh sca-training-vm --zone=us-central1-a -- -L 6006:localhost:6006

# Open in browser: http://localhost:6006
```

### Method 2: Check Logs

```bash
# View real-time training logs
tail -f training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check checkpoint files
ls -lh sca-package-checkpoints/
```

### Method 3: Cloud Console

1. Go to **Compute Engine** → **VM Instances**
2. Click on `sca-training-vm`
3. Click **Monitoring** tab
4. View CPU, GPU, disk, network usage

---

## Part 7: Download Trained Model

### Method 1: Cloud Storage (Recommended)

```bash
# Install gsutil (if not already installed)
sudo apt-get install -y google-cloud-sdk

# Create a Cloud Storage bucket
gsutil mb -l us-central1 gs://ai-sec-models-$(date +%s)

# Upload trained model
gsutil -m cp -r ./sca-package-final gs://ai-sec-models-YOUR_BUCKET_NAME/

# Download to your local machine:
# On your LOCAL computer:
gsutil -m cp -r gs://ai-sec-models-YOUR_BUCKET_NAME/sca-package-final ./
```

### Method 2: Direct Download via SCP

```bash
# On your LOCAL machine:
gcloud compute scp --recurse sca-training-vm:~/ai_sec/sca-package-final ./sca-package-final --zone=us-central1-a
```

### Method 3: Zip and Download

```bash
# On the VM:
cd ~/ai_sec
tar -czf sca-package-final.tar.gz sca-package-final/

# Download on your LOCAL machine:
gcloud compute scp sca-training-vm:~/ai_sec/sca-package-final.tar.gz ./ --zone=us-central1-a
```

---

## Part 8: Stop/Delete VM to Save Costs

### Option A: Stop VM (Resume later)

```bash
# Stop VM (keeps disk, no compute charges, only storage ~$4/month for 100GB)
gcloud compute instances stop sca-training-vm --zone=us-central1-a

# Start later:
gcloud compute instances start sca-training-vm --zone=us-central1-a
```

### Option B: Delete VM (Completely remove)

```bash
# Delete VM (saves all costs, but you'll lose everything on disk)
gcloud compute instances delete sca-training-vm --zone=us-central1-a

# Confirm: Type 'y' and press Enter
```

**💡 Tip**: Always **stop** the VM after training completes, don't leave it running!

---

## Part 9: Cost Optimization Tips

### 1. Use Preemptible/Spot Instances (70% cheaper!)

```bash
# When creating VM, add this flag:
gcloud compute instances create sca-training-vm \
  --preemptible \
  # ... rest of the flags ...
```

**Savings**: $0.54/hour → $0.16/hour (~$1 total for 6 hours)

**Trade-off**: Can be interrupted after 24 hours (saves checkpoints every 100 steps, so safe to resume)

### 2. Use Automatic Shutdown

```bash
# Add shutdown script to VM metadata:
gcloud compute instances add-metadata sca-training-vm \
  --zone=us-central1-a \
  --metadata=shutdown-script='#!/bin/bash
echo "Training complete, auto-shutdown in 5 minutes..."
sleep 300
sudo shutdown -h now'
```

### 3. Monitor Spending

1. Go to **Billing** → **Budget & Alerts**
2. Create budget: Set $10 limit
3. Enable alerts at 50%, 80%, 100%

---

## Troubleshooting

### Issue 1: "Quota exceeded" error

**Solution**: Wait for quota increase approval (check email), or try different region

### Issue 2: GPU not detected

```bash
# Reinstall NVIDIA drivers
sudo /opt/deeplearning/install-driver.sh
sudo reboot

# After reboot, check again:
nvidia-smi
```

### Issue 3: Out of memory error

**Solution**: Reduce batch size in training script:
```python
per_device_train_batch_size=1  # Already at minimum
gradient_accumulation_steps=8  # Reduce from 16 to 8
```

### Issue 4: Training too slow

**Upgrade to better GPU**:
- V100: 3x faster (~$2.67/hour, ~2 hours training = $5.50 total)
- A100: 5x faster (~$3.67/hour, ~1 hour training = $3.70 total)

### Issue 5: SSH connection lost

```bash
# Training continues if you used tmux!
# Reconnect:
gcloud compute ssh sca-training-vm --zone=us-central1-a

# Reattach to tmux session:
tmux attach -t training
```

---

## Quick Reference Commands

```bash
# Create VM with T4 GPU
gcloud compute instances create sca-training-vm --zone=us-central1-a --machine-type=n1-standard-4 --accelerator=type=nvidia-tesla-t4,count=1 --image-family=common-cu118-debian-11-py310 --image-project=deeplearning-platform-release --boot-disk-size=100GB --metadata="install-nvidia-driver=True"

# SSH into VM
gcloud compute ssh sca-training-vm --zone=us-central1-a

# Check GPU
nvidia-smi

# Run training in background
tmux new -s training
python3 train_sca_gcp.py 2>&1 | tee training.log
# Detach: Ctrl+B, then D

# Monitor progress
tail -f training.log
tmux attach -t training

# Stop VM when done
gcloud compute instances stop sca-training-vm --zone=us-central1-a

# Delete VM
gcloud compute instances delete sca-training-vm --zone=us-central1-a
```

---

## Summary

**Total Cost Estimate**:
- **Free Tier**: $0 (use $300 free credits)
- **Regular T4**: ~$3 for 6 hours
- **Preemptible T4**: ~$1 for 6 hours
- **V100**: ~$5.50 for 2 hours (faster)

**Time Required**:
- Setup: 15 minutes (first time)
- Training: 4-6 hours (T4), 1.5-2 hours (V100)
- Download: 5-10 minutes

**Next Steps**:
1. Complete training
2. Download model
3. Deploy using vLLM or Ollama (see [04_MODEL_DEPLOYMENT.md](04_MODEL_DEPLOYMENT.md))
4. Integrate into CI/CD pipeline

---

Need help? Check the [GCP Documentation](https://cloud.google.com/compute/docs/gpus) or ask in the chat!
