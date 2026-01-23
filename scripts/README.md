# Training Scripts

This folder contains all the scripts needed to train your SCA model.

## Quick Start for Vertex AI Training

### 1️⃣ Get Your GCP Project Info

Run this first to find or create your GCP project:

```bash
./get_gcp_info.sh
```

This interactive script will:
- Help you login to Google Cloud
- Show your existing projects or create a new one
- Give you your **Project ID** and **Bucket Name**
- Save the configuration for later use

### 2️⃣ Run GCP Setup (One-time)

```bash
./setup_gcp_vertex.sh
```

This will:
- Enable required APIs
- Create Cloud Storage bucket
- Set up TensorBoard
- Show you how to request GPU quota

### 3️⃣ Request GPU Quota

Follow the instructions shown by the setup script. Takes 5-30 minutes for approval.

### 4️⃣ Submit Training Job

Once GPU quota is approved:

```bash
./submit_vertex_training.sh YOUR_PROJECT_ID
```

Example:
```bash
./submit_vertex_training.sh ai-sec-training-123456
```

---

## Available Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `get_gcp_info.sh` | Find/create GCP project and get IDs | **Start here** - First time setup |
| `setup_gcp_vertex.sh` | Configure GCP for Vertex AI | Once per project |
| `submit_vertex_training.sh` | Submit training job to Vertex AI | Every time you want to train |
| `train_sca_local.py` | Train locally (requires GPU) | If you have local GPU |
| `vertex_train_sca.py` | Vertex AI training script | Don't run directly (used by submit script) |

---

## Complete Workflow

```bash
# Step 1: Get your GCP info
cd /workspaces/ai_sec/scripts
./get_gcp_info.sh
# Follow prompts, note down your Project ID

# Step 2: Setup GCP (one-time)
./setup_gcp_vertex.sh
# When prompted, enter your Project ID

# Step 3: Request GPU quota
# Follow the link shown in setup script output
# Wait for approval email (5-30 mins)

# Step 4: Submit training job
./submit_vertex_training.sh your-project-id-123456

# Step 5: Monitor training
# Use the links shown in the output
```

---

## File Descriptions

### Training Scripts

**`vertex_train_sca.py`**
- Main training script optimized for Vertex AI
- Uses CodeLlama 7B with LoRA fine-tuning
- Automatically handles checkpointing and logging
- Configured for T4 GPU (16GB VRAM)

**`train_sca_local.py`**
- Alternative for local GPU training
- Same model configuration
- Use if you have your own GPU

### Configuration Files

**`requirements-vertex.txt`**
- Python dependencies for Vertex AI training
- Includes PyTorch, Transformers, PEFT, etc.

### Helper Scripts

**`get_gcp_info.sh`**
- Interactive helper to get your Project ID and Bucket Name
- Creates projects if needed
- Saves configuration to `.gcp_config`

**`setup_gcp_vertex.sh`**
- One-time GCP project setup
- Enables APIs, creates bucket, sets up TensorBoard
- Shows GPU quota request instructions

**`submit_vertex_training.sh`**
- Uploads code to Cloud Storage
- Submits training job to Vertex AI
- Shows monitoring links

---

## Cost Estimates

### Vertex AI Training (Recommended)

- **GPU**: T4 (~$0.54/hour)
- **Training time**: 4-6 hours
- **Total cost**: ~$2-4 per run
- **Free credits**: $300 for new users

### Local Training

- **Requires**: NVIDIA GPU with 16GB+ VRAM
- **Time**: 4-6 hours
- **Cost**: Your electricity bill 😊

---

## Troubleshooting

### "gcloud: command not found"

gcloud is now installed! Run:
```bash
source ~/google-cloud-sdk/path.bash.inc
```

Or just re-run `get_gcp_info.sh` which will install it.

### "Project not found"

Run:
```bash
gcloud config set project YOUR_PROJECT_ID
```

### "Quota exceeded" for GPU

You need to request GPU quota. See [GCP_SETUP_GUIDE.md](../GCP_SETUP_GUIDE.md) Step 8.

### "Permission denied"

Authenticate again:
```bash
gcloud auth login
gcloud auth application-default login
```

---

## Monitoring Your Training

### Web Console

View jobs at:
```
https://console.cloud.google.com/vertex-ai/training?project=YOUR_PROJECT_ID
```

### Command Line

```bash
# List all jobs
gcloud ai custom-jobs list --region=us-central1

# Stream logs
gcloud ai custom-jobs stream-logs JOB_NAME --region=us-central1

# Job status
gcloud ai custom-jobs describe JOB_NAME --region=us-central1
```

### TensorBoard

View metrics at:
```
https://console.cloud.google.com/vertex-ai/experiments?project=YOUR_PROJECT_ID
```

---

## Download Trained Model

After training completes:

```bash
# Download to local directory
gsutil -m cp -r gs://YOUR_BUCKET/models/sca-*/model/ ./trained_model/

# Example:
gsutil -m cp -r gs://ai-sec-training-123456-ml-data/models/sca-*/model/ ./trained_model/
```

---

## Need Help?

1. **Full setup guide**: See [GCP_SETUP_GUIDE.md](../GCP_SETUP_GUIDE.md)
2. **Vertex AI details**: See [docs/08_VERTEX_AI_TRAINING.md](../docs/08_VERTEX_AI_TRAINING.md)
3. **GCP Console**: https://console.cloud.google.com

---

## What's Next?

After training:
1. Download your model
2. Test it locally
3. Deploy to production

See [docs/04_MODEL_DEPLOYMENT.md](../docs/04_MODEL_DEPLOYMENT.md) for deployment instructions.
