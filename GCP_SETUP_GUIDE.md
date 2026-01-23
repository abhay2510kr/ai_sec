# Google Cloud Platform Setup Guide
## Getting Your Project ID and Bucket Name

This guide will walk you through creating a GCP account and getting the information needed for Vertex AI training.

---

## Part 1: Create GCP Account

### Step 1: Sign Up for Google Cloud

1. Go to: **https://cloud.google.com/free**
2. Click **"Get started for free"**
3. Sign in with your Google account (or create one)
4. Enter your:
   - Country
   - Payment information (required but you won't be charged)
   - Accept terms and conditions
5. Click **"Start my free trial"**

**You get $300 free credits valid for 90 days!**

---

## Part 2: Create Your First Project

### Step 2: Create a Project

Once you're in the GCP Console:

1. Go to: **https://console.cloud.google.com**
2. Click the project dropdown at the top (it might say "My First Project" or "Select a project")
3. Click **"NEW PROJECT"**
4. Fill in:
   - **Project name**: `ai-sec-training` (or your preferred name)
   - **Organization**: Leave as "No organization" (unless you have one)
5. Click **"CREATE"**

### Step 3: Get Your Project ID

After creating the project:

1. The **Project ID** will be auto-generated (e.g., `ai-sec-training-123456`)
2. Note this down - **THIS IS YOUR PROJECT_ID**
3. You can always find it:
   - Top bar of GCP Console
   - Dashboard page
   - Project settings

**Example:**
```
Project Name: ai-sec-training
Project ID: ai-sec-training-123456   ← USE THIS
```

---

## Part 3: Understanding Bucket Names

### What is a Cloud Storage Bucket?

Think of it as a folder in the cloud where you'll store:
- Training code
- Model checkpoints
- Final trained models

### Your Bucket Name

The bucket name follows this pattern:
```
[PROJECT_ID]-ml-data
```

**Example:**
- If your Project ID is `ai-sec-training-123456`
- Your bucket name will be: `ai-sec-training-123456-ml-data`

**The script will create this bucket automatically!** You don't need to create it manually.

---

## Part 4: Authentication Setup

### Step 4: Authenticate with Google Cloud

From your terminal:

```bash
# Login to Google Cloud
gcloud auth login
```

This will:
1. Open your browser
2. Ask you to sign in with your Google account
3. Grant permissions to gcloud CLI

### Step 5: Set Your Project

```bash
# Set your active project
gcloud config set project YOUR_PROJECT_ID

# Example:
gcloud config set project ai-sec-training-123456
```

### Step 6: Enable Billing

1. Go to: **https://console.cloud.google.com/billing**
2. Link your project to the billing account (uses free credits first)
3. This is required to use GPUs

---

## Part 5: Run the Setup Script

### Step 7: Initial Setup

Now run the setup script:

```bash
cd /workspaces/ai_sec/scripts
./setup_gcp_vertex.sh
```

When prompted:
- **Enter your GCP Project ID**: `ai-sec-training-123456` (your actual ID)
- **Create TensorBoard?**: `y` (recommended)

The script will:
1. Enable required APIs (Vertex AI, Compute Engine, etc.)
2. Create Cloud Storage bucket
3. Set up TensorBoard
4. Show instructions for GPU quota

### Step 8: Request GPU Quota

**IMPORTANT:** By default, new projects have 0 GPU quota. You need to request it.

1. Go to: https://console.cloud.google.com/iam-admin/quotas
2. In the filter:
   - **Service**: Type "Vertex AI API"
   - **Metric**: Type "NVIDIA_TESLA_T4"
   - **Location**: Select "us-central1"
3. Select the checkbox for the quota
4. Click **"EDIT QUOTAS"** at the top
5. Enter new limit: **1**
6. Add justification: *"Training machine learning models for security vulnerability detection"*
7. Click **"SUBMIT REQUEST"**

**Approval time:** Usually 5-30 minutes for T4 GPUs

You'll receive an email when approved.

---

## Part 6: Submit Your Training Job

### Step 9: Wait for GPU Quota Approval

Check your email for approval. You can also check status:

```bash
gcloud compute project-info describe --project=YOUR_PROJECT_ID
```

### Step 10: Submit Training Job

Once quota is approved:

```bash
cd /workspaces/ai_sec/scripts
./submit_vertex_training.sh YOUR_PROJECT_ID
```

**Example:**
```bash
./submit_vertex_training.sh ai-sec-training-123456
```

The script will automatically:
1. Use the project ID you provide
2. Create bucket name as: `ai-sec-training-123456-ml-data`
3. Upload training code
4. Submit the training job
5. Show you monitoring links

---

## Quick Reference

### Finding Your Information

**Project ID:**
```bash
gcloud config get-value project
```

**List all your projects:**
```bash
gcloud projects list
```

**Check if bucket exists:**
```bash
gsutil ls
```

**View your quota:**
```bash
gcloud compute project-info describe --project=YOUR_PROJECT_ID | grep NVIDIA
```

---

## Monitoring Your Training Job

### View Jobs in Console

https://console.cloud.google.com/vertex-ai/training?project=YOUR_PROJECT_ID

### Stream Logs in Terminal

```bash
# List all jobs
gcloud ai custom-jobs list --region=us-central1

# Stream logs for specific job
gcloud ai custom-jobs stream-logs JOB_NAME --region=us-central1
```

### View TensorBoard

https://console.cloud.google.com/vertex-ai/experiments?project=YOUR_PROJECT_ID

---

## Cost Tracking

### View Current Costs

1. Go to: https://console.cloud.google.com/billing
2. Click your billing account
3. View **"Reports"** tab

### Set Budget Alert

1. Go to: https://console.cloud.google.com/billing/budgets
2. Click **"CREATE BUDGET"**
3. Set:
   - Name: "Training Budget"
   - Amount: $50 (or your limit)
   - Alert threshold: 50%, 90%, 100%
4. You'll receive email alerts when reaching thresholds

---

## Troubleshooting

### Issue: "Project not found"

**Solution:**
```bash
gcloud config set project YOUR_ACTUAL_PROJECT_ID
```

### Issue: "Quota exceeded" error

**Solution:** You need to wait for GPU quota approval (Step 8)

### Issue: "Permission denied"

**Solution:**
```bash
gcloud auth login
gcloud auth application-default login
```

### Issue: "Bucket already exists"

**Solution:** Someone else might be using that name. Change your project ID or bucket name.

---

## Summary Checklist

- [ ] Created GCP account ($300 free credits)
- [ ] Created project (got Project ID)
- [ ] Authenticated with `gcloud auth login`
- [ ] Set project with `gcloud config set project PROJECT_ID`
- [ ] Ran `./setup_gcp_vertex.sh`
- [ ] Requested T4 GPU quota
- [ ] Received quota approval email
- [ ] Ready to run `./submit_vertex_training.sh PROJECT_ID`

---

## Example Complete Workflow

```bash
# 1. Authenticate
gcloud auth login

# 2. Set project
gcloud config set project ai-sec-training-123456

# 3. Run setup (one-time)
cd /workspaces/ai_sec/scripts
./setup_gcp_vertex.sh

# 4. Wait for GPU quota approval (check email)
# ... wait 5-30 minutes ...

# 5. Submit training job
./submit_vertex_training.sh ai-sec-training-123456

# 6. Monitor
gcloud ai custom-jobs list --region=us-central1
```

---

## Need Help?

### GCP Documentation
- **Quickstart**: https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform
- **Pricing**: https://cloud.google.com/vertex-ai/pricing
- **Free tier**: https://cloud.google.com/free

### Support
- **GCP Console**: https://console.cloud.google.com/support
- **Community**: https://groups.google.com/g/google-cloud-dev

---

**Next Steps:** After training completes, see [docs/04_MODEL_DEPLOYMENT.md](docs/04_MODEL_DEPLOYMENT.md) for deployment instructions.
