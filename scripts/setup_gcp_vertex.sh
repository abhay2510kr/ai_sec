#!/bin/bash
#
# One-time GCP Vertex AI setup script
# Run this once to configure your GCP project for Vertex AI training
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "================================================================================================"
echo "🔧 Google Cloud Vertex AI - Initial Setup"
echo "================================================================================================"

# Get project ID
read -p "Enter your GCP Project ID (or press Enter for 'ai-sec-training'): " PROJECT_ID
PROJECT_ID="${PROJECT_ID:-ai-sec-training}"

REGION="us-central1"
BUCKET_NAME="${PROJECT_ID}-ml-data"

echo ""
echo "📋 Configuration:"
echo "  - Project ID: ${PROJECT_ID}"
echo "  - Region: ${REGION}"
echo "  - Bucket: ${BUCKET_NAME}"
echo ""

# Check gcloud
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Error: gcloud CLI not installed${NC}"
    echo ""
    echo "Install it from: https://cloud.google.com/sdk/docs/install"
    echo ""
    echo "Or use Cloud Shell: https://console.cloud.google.com"
    exit 1
fi

echo -e "${GREEN}✅ gcloud CLI found${NC}"
echo ""

# Authenticate
echo "🔐 Authenticating with Google Cloud..."
gcloud auth login
gcloud config set project ${PROJECT_ID}

# Enable APIs
echo ""
echo "🔌 Enabling required APIs..."
echo "  This may take a few minutes..."
echo ""

gcloud services enable aiplatform.googleapis.com
echo -e "${GREEN}  ✅ Vertex AI API${NC}"

gcloud services enable compute.googleapis.com
echo -e "${GREEN}  ✅ Compute Engine API${NC}"

gcloud services enable storage-api.googleapis.com
echo -e "${GREEN}  ✅ Cloud Storage API${NC}"

gcloud services enable artifactregistry.googleapis.com
echo -e "${GREEN}  ✅ Artifact Registry API${NC}"

# Create bucket
echo ""
echo "📦 Creating Cloud Storage bucket..."
if gsutil ls gs://${BUCKET_NAME} &> /dev/null; then
    echo -e "${YELLOW}  ⚠️  Bucket already exists${NC}"
else
    gsutil mb -l ${REGION} gs://${BUCKET_NAME}
    gsutil mkdir gs://${BUCKET_NAME}/datasets/
    gsutil mkdir gs://${BUCKET_NAME}/models/
    gsutil mkdir gs://${BUCKET_NAME}/training-code/
    echo -e "${GREEN}  ✅ Bucket created: gs://${BUCKET_NAME}${NC}"
fi

# Check GPU quota
echo ""
echo "================================================================================================"
echo "🎯 GPU Quota Check"
echo "================================================================================================"
echo ""
echo "To use GPUs for training, you need to request quota increase."
echo ""
echo -e "${YELLOW}📝 Steps to request GPU quota:${NC}"
echo ""
echo "  1. Go to: https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT_ID}"
echo ""
echo "  2. Filter by:"
echo "     - Service: 'Vertex AI API'"
echo "     - Metric: 'Custom model training NVIDIA_TESLA_T4 GPUs per region'"
echo "     - Location: '${REGION}'"
echo ""
echo "  3. Select the quota and click 'EDIT QUOTAS'"
echo ""
echo "  4. Request new limit: 1 (or more)"
echo ""
echo "  5. Provide justification: 'Training security ML models for vulnerability detection'"
echo ""
echo "  6. Submit request"
echo ""
echo "  ⏱️  Approval usually takes 5-30 minutes for T4 GPUs"
echo ""

# Create TensorBoard
echo "================================================================================================"
echo "📊 TensorBoard Setup"
echo "================================================================================================"
echo ""

read -p "Create TensorBoard instance for monitoring? (y/n): " CREATE_TB

if [ "$CREATE_TB" = "y" ]; then
    echo "Creating TensorBoard instance..."
    TB_NAME="sca-training-metrics"
    
    if gcloud ai tensorboards list --region=${REGION} --filter="displayName:${TB_NAME}" --format="value(name)" | grep -q "tensorboards"; then
        echo -e "${YELLOW}  ⚠️  TensorBoard instance already exists${NC}"
    else
        gcloud ai tensorboards create \
          --display-name="${TB_NAME}" \
          --region=${REGION}
        echo -e "${GREEN}  ✅ TensorBoard created${NC}"
    fi
    
    echo ""
    echo "View TensorBoard at:"
    echo "https://console.cloud.google.com/vertex-ai/experiments/tensorboard?project=${PROJECT_ID}"
fi

# Summary
echo ""
echo "================================================================================================"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "================================================================================================"
echo ""
echo "📋 Summary:"
echo ""
echo "  ✅ Project configured: ${PROJECT_ID}"
echo "  ✅ Required APIs enabled"
echo "  ✅ Storage bucket created: gs://${BUCKET_NAME}"
echo ""
echo "⚠️  Next Steps:"
echo ""
echo "  1. Request GPU quota (see instructions above)"
echo ""
echo "  2. Once quota is approved, submit training job:"
echo "     cd /workspaces/ai_sec/scripts"
echo "     chmod +x submit_vertex_training.sh"
echo "     ./submit_vertex_training.sh ${PROJECT_ID} ${BUCKET_NAME}"
echo ""
echo "  3. Monitor training:"
echo "     https://console.cloud.google.com/vertex-ai/training?project=${PROJECT_ID}"
echo ""
echo "================================================================================================"
echo ""
echo "💡 Estimated Costs:"
echo "  - T4 GPU training: ~\$0.54/hour"
echo "  - Expected training time: 4-6 hours"
echo "  - Total cost: ~\$2-4 per training run"
echo ""
echo "  ✨ New GCP users get \$300 free credits!"
echo ""
echo "================================================================================================"
