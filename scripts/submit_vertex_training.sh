#!/bin/bash
#
# Submit training job to Google Cloud Vertex AI
# Usage: ./submit_vertex_training.sh [PROJECT_ID] [BUCKET_NAME]
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================================================"
echo "🚀 Vertex AI Training Job Submission"
echo "================================================================================================"

# Configuration
PROJECT_ID="${1:-ai-sec-training}"
BUCKET_NAME="${2:-${PROJECT_ID}-ml-data}"
REGION="us-central1"
JOB_NAME="sca-training-$(date +%Y%m%d-%H%M%S)"
MACHINE_TYPE="n1-standard-4"
ACCELERATOR="NVIDIA_TESLA_T4"
ACCELERATOR_COUNT="1"

echo ""
echo "📋 Configuration:"
echo "  - Project ID: ${PROJECT_ID}"
echo "  - Bucket: ${BUCKET_NAME}"
echo "  - Region: ${REGION}"
echo "  - Machine: ${MACHINE_TYPE}"
echo "  - GPU: ${ACCELERATOR} x${ACCELERATOR_COUNT}"
echo "  - Job Name: ${JOB_NAME}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Error: gcloud CLI not found${NC}"
    echo "Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set project
echo "🔧 Setting GCP project..."
gcloud config set project ${PROJECT_ID}

# Check if bucket exists, create if not
echo "📦 Checking Cloud Storage bucket..."
if ! gsutil ls gs://${BUCKET_NAME} &> /dev/null; then
    echo "Creating bucket: gs://${BUCKET_NAME}"
    gsutil mb -l ${REGION} gs://${BUCKET_NAME}
    gsutil mkdir gs://${BUCKET_NAME}/datasets/
    gsutil mkdir gs://${BUCKET_NAME}/models/
    gsutil mkdir gs://${BUCKET_NAME}/training-code/
    echo -e "${GREEN}✅ Bucket created${NC}"
else
    echo -e "${GREEN}✅ Bucket exists${NC}"
fi

# Upload training code
echo ""
echo "📤 Uploading training code to Cloud Storage..."
gsutil cp vertex_train_sca.py gs://${BUCKET_NAME}/training-code/
gsutil cp requirements-vertex.txt gs://${BUCKET_NAME}/training-code/requirements.txt
echo -e "${GREEN}✅ Code uploaded${NC}"

# Create training job config
echo ""
echo "🚀 Submitting training job..."
echo ""

# Create a simple training script wrapper
cat > /tmp/train_wrapper.sh << 'WRAPPER_EOF'
#!/bin/bash
set -e
cd /tmp
gsutil cp gs://BUCKET_PLACEHOLDER/training-code/vertex_train_sca.py ./
gsutil cp gs://BUCKET_PLACEHOLDER/training-code/requirements.txt ./
pip install -q -r requirements.txt
python vertex_train_sca.py
WRAPPER_EOF

# Replace placeholder
sed -i "s|BUCKET_PLACEHOLDER|${BUCKET_NAME}|g" /tmp/train_wrapper.sh

# Upload wrapper
gsutil cp /tmp/train_wrapper.sh gs://${BUCKET_NAME}/training-code/

# Submit job
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${JOB_NAME} \
  --worker-pool-spec="machine-type=${MACHINE_TYPE},replica-count=1,accelerator-type=${ACCELERATOR},accelerator-count=${ACCELERATOR_COUNT},container-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest" \
  --args="bash,-c,gsutil cp gs://${BUCKET_NAME}/training-code/train_wrapper.sh /tmp/ && chmod +x /tmp/train_wrapper.sh && /tmp/train_wrapper.sh" \
  --enable-web-access

echo ""
echo "================================================================================================"
echo -e "${GREEN}✅ Training job submitted successfully!${NC}"
echo "================================================================================================"
echo ""
echo "📊 Monitor your job:"
echo ""
echo "  1. Web Console:"
echo "     https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo ""
echo "  2. List jobs:"
echo "     gcloud ai custom-jobs list --region=${REGION}"
echo ""
echo "  3. Stream logs:"
echo "     gcloud ai custom-jobs stream-logs ${JOB_NAME} --region=${REGION}"
echo ""
echo "  4. Job details:"
echo "     gcloud ai custom-jobs describe ${JOB_NAME} --region=${REGION}"
echo ""
echo "💾 After training completes, download your model:"
echo "     gsutil -m cp -r gs://${BUCKET_NAME}/models/sca-*/model/ ./trained_model/"
echo ""
echo "================================================================================================"
