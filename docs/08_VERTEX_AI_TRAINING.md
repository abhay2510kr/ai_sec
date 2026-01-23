# Vertex AI Training Setup Guide

Complete guide to train your SCA model using Google Cloud Vertex AI - **the recommended approach** instead of managing VMs.

---

## Why Vertex AI vs VM?

| Feature | Vertex AI | Manual VM |
|---------|-----------|-----------|
| **Setup** | Managed, pre-configured | Manual CUDA/driver setup |
| **Scaling** | Auto-scale GPUs | Manual resizing |
| **Monitoring** | Built-in dashboards | Manual setup |
| **Cost** | Pay per training job | Pay while VM runs |
| **Deployment** | One-click deploy | Manual setup |
| **Experiment Tracking** | Automatic | Manual MLflow setup |

---

## Prerequisites

- Google Cloud account ([Create one](https://cloud.google.com))
- $300 free credits (new users)
- Your training data in Cloud Storage or GitHub

---

## Part 1: Initial Setup (One-time)

### Step 1: Create Project & Enable APIs

```bash
# Open Cloud Shell: https://console.cloud.google.com

# Set project name
export PROJECT_ID="ai-sec-training"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### Step 2: Request GPU Quota

1. Go to: **IAM & Admin** → **Quotas**
2. Filter: 
   - Service: "Vertex AI API"
   - Metric: "Custom model training NVIDIA_TESLA_T4 GPUs per region"
3. Select your region (e.g., `us-central1`)
4. Click **Edit Quotas** → Request limit: `1`
5. Submit (usually approved in 5-30 minutes)

### Step 3: Create Storage Bucket

```bash
# Create bucket for models and data
export BUCKET_NAME="${PROJECT_ID}-ml-data"
gsutil mb -l us-central1 gs://${BUCKET_NAME}

# Create directories
gsutil mkdir gs://${BUCKET_NAME}/datasets/
gsutil mkdir gs://${BUCKET_NAME}/models/
gsutil mkdir gs://${BUCKET_NAME}/training-code/
```

---

## Part 2: Prepare Training Code

### Option A: Use Pre-configured Container (Recommended)

Create a training script that Vertex AI can run:

**`vertex_train_sca.py`**:

```python
#!/usr/bin/env python3
"""Vertex AI training script for SCA model"""

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
    print("🚀 Starting Vertex AI Training Job")
    print(f"📊 TensorBoard logs: {TENSORBOARD_LOG_DIR}")
    print(f"💾 Model output: {MODEL_DIR}")
    
    # Check GPU
    print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load dataset
    print("\n📥 Loading dataset...")
    dataset = load_dataset('json', data_files={'train': DATASET_URL})
    train_dataset = dataset['train']
    print(f"✅ Loaded {len(train_dataset)} training examples")
    
    # Load tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )
    
    print("🔄 Tokenizing dataset...")
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    # Load model with quantization
    print("\n🤖 Loading model with 4-bit quantization...")
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
        trust_remote_code=True
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print("⚙️  Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        fp16=True,
        save_strategy="epoch",
        logging_dir=TENSORBOARD_LOG_DIR,
        logging_steps=10,
        report_to="tensorboard",
        save_total_limit=2,
        warmup_steps=100,
        lr_scheduler_type="cosine"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    
    # Train
    print("\n🏋️  Starting training...")
    trainer.train()
    
    # Save final model
    print(f"\n💾 Saving model to {MODEL_DIR}...")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    
    print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main()
```

**Upload training code to Cloud Storage**:

```bash
# Upload your script
gsutil cp vertex_train_sca.py gs://${BUCKET_NAME}/training-code/

# Create requirements.txt
cat > requirements.txt << 'EOF'
torch==2.1.2
transformers==4.37.0
datasets==2.16.0
accelerate==0.26.0
peft==0.8.0
bitsandbytes==0.42.0
sentencepiece
protobuf
google-cloud-storage
EOF

gsutil cp requirements.txt gs://${BUCKET_NAME}/training-code/
```

---

## Part 3: Launch Training Job

### Option A: Using Console (Beginner-friendly)

1. **Go to Vertex AI** → **Training** → **Create Training Job**

2. **Dataset**: No managed dataset (we load from GitHub)

3. **Model Training**:
   - Method: `Custom training`
   - Container: `Pre-built container`
   
4. **Model details**:
   - Model name: `sca-vulnerability-model`
   - Container image: `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest`
   - Model output directory: `gs://YOUR_BUCKET/models/sca/`

5. **Training container**:
   - Package location: `gs://YOUR_BUCKET/training-code/`
   - Python module: `vertex_train_sca`

6. **Compute**:
   - Region: `us-central1`
   - Machine type: `n1-standard-4`
   - Accelerator: `NVIDIA_TESLA_T4`
   - Count: `1`

7. **Click "START TRAINING"**

### Option B: Using gcloud CLI (Recommended)

```bash
# Set variables
export REGION="us-central1"
export JOB_NAME="sca_training_$(date +%Y%m%d_%H%M%S)"
export MACHINE_TYPE="n1-standard-4"
export ACCELERATOR="NVIDIA_TESLA_T4"

# Submit training job
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${JOB_NAME} \
  --worker-pool-spec=machine-type=${MACHINE_TYPE},replica-count=1,accelerator-type=${ACCELERATOR},accelerator-count=1,container-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest,local-package-path=.,script=vertex_train_sca.py \
  --args="--bucket=${BUCKET_NAME}"
```

### Option C: Using Python SDK (Advanced)

```python
from google.cloud import aiplatform

# Initialize
aiplatform.init(project='ai-sec-training', location='us-central1')

# Create custom training job
job = aiplatform.CustomContainerTrainingJob(
    display_name="sca-training",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest",
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest"
)

# Run the job
model = job.run(
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    base_output_dir=f"gs://{BUCKET_NAME}/models/sca/",
    tensorboard=f"projects/{PROJECT_ID}/locations/us-central1/tensorboards/YOUR_TENSORBOARD_ID"
)
```

---

## Part 4: Monitor Training

### TensorBoard Integration

```bash
# Create TensorBoard instance (one-time)
gcloud ai tensorboards create \
  --display-name="sca-training-metrics" \
  --region=us-central1

# Get TensorBoard URL
gcloud ai tensorboards list --region=us-central1
```

Then view in browser: **Vertex AI** → **Experiments** → Select your TensorBoard

### Check Job Status

```bash
# List jobs
gcloud ai custom-jobs list --region=us-central1

# Get job details
gcloud ai custom-jobs describe JOB_ID --region=us-central1

# Stream logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

---

## Part 5: Download & Deploy Model

### Download Trained Model

```bash
# Download from Cloud Storage
gsutil -m cp -r gs://${BUCKET_NAME}/models/sca/latest/ ./trained_model/

# Or download specific files
gsutil cp gs://${BUCKET_NAME}/models/sca/latest/adapter_model.bin ./
gsutil cp gs://${BUCKET_NAME}/models/sca/latest/adapter_config.json ./
```

### Deploy to Vertex AI Endpoint (Optional)

```python
from google.cloud import aiplatform

# Upload model
model = aiplatform.Model.upload(
    display_name="sca-vulnerability-detector",
    artifact_uri=f"gs://{BUCKET_NAME}/models/sca/latest/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest"
)

# Create endpoint
endpoint = aiplatform.Endpoint.create(display_name="sca-inference")

# Deploy model
endpoint.deploy(
    model=model,
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=3
)

# Make prediction
prediction = endpoint.predict(instances=[{
    "text": "Analyze this package.json: ..."
}])
```

---

## Pricing Comparison

### Vertex AI Custom Training (Recommended)

**T4 GPU**: ~$0.54/hour
- Training time: ~6 hours
- **Total cost**: ~$3.24

**Advantages**:
- ✅ Only pay during training
- ✅ No idle VM costs
- ✅ Automatic shutdown
- ✅ Built-in monitoring

### Manual VM (Not Recommended)

**n1-standard-4 + T4**: ~$0.54/hour
- Setup time: ~1 hour
- Training: ~6 hours  
- Cleanup: ~0.5 hours
- **Total cost**: ~$4.05 (if you remember to shut it down!)

**Risks**:
- ❌ Forget to shutdown = $$$ wasted
- ❌ Manual monitoring
- ❌ Manual setup

---

## Troubleshooting

### Quota Error

```
Error: Quota 'NVIDIA_T4_GPUS' exceeded
```

**Solution**: Request GPU quota increase (see Step 2 in Part 1)

### Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size or increase gradient accumulation:

```python
BATCH_SIZE = 2  # Reduce from 4
GRADIENT_ACCUMULATION = 8  # Increase from 4
```

### Container Not Found

```
Error: Container image not found
```

**Solution**: Use correct PyTorch container:

```bash
us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest
```

---

## Advanced: Hyperparameter Tuning

Run multiple experiments automatically:

```python
from google.cloud import aiplatform

# Define hyperparameter spec
hparam_spec = {
    'parameter_spec': {
        'learning_rate': aiplatform.gapic.StudySpec.ParameterSpec.DoubleValueSpec(
            min_value=1e-5, max_value=1e-3
        ),
        'batch_size': aiplatform.gapic.StudySpec.ParameterSpec.DiscreteValueSpec(
            values=[2, 4, 8]
        )
    }
}

# Create tuning job
job = aiplatform.HyperparameterTuningJob(
    display_name='sca-hparam-tuning',
    custom_job=custom_job_spec,
    metric_spec={'loss': 'minimize'},
    parameter_spec=hparam_spec,
    max_trial_count=10,
    parallel_trial_count=2
)

job.run()
```

---

## Best Practices

1. **Start with small experiments** (1 epoch, 10% data) to verify setup
2. **Use TensorBoard** for real-time monitoring
3. **Enable checkpointing** to resume interrupted training
4. **Use pre-emptible VMs** for 60-70% cost savings (for non-urgent jobs)
5. **Clean up resources** after training completes

---

## Quick Start Command

Complete end-to-end training:

```bash
# 1. Set project
gcloud config set project ai-sec-training

# 2. Submit job
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=sca-training-$(date +%s) \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest,python-module=vertex_train_sca \
  --python-package-uris=gs://YOUR_BUCKET/training-code/vertex_train_sca.py

# 3. Monitor
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1

# 4. Download model
gsutil -m cp -r gs://YOUR_BUCKET/models/sca/latest/ ./trained_model/
```

---

## Next Steps

- [Model Deployment Guide](04_MODEL_DEPLOYMENT.md)
- [Integration & Orchestration](05_INTEGRATION_ORCHESTRATION.md)

**💡 Tip**: Use Vertex AI Workbench (managed Jupyter) for interactive development before submitting training jobs.
