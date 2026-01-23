#!/bin/bash
#
# Interactive helper script to guide you through GCP setup
# This makes it easy to get your Project ID and Bucket Name
#

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

clear
echo "================================================================================================"
echo "🎯 GCP Project & Bucket Information Helper"
echo "================================================================================================"
echo ""
echo "This script will help you find or create your GCP Project ID and Bucket Name."
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${YELLOW}⚠️  gcloud CLI not found. Installing...${NC}"
    echo ""
    curl https://sdk.cloud.google.com | bash
    source ~/google-cloud-sdk/path.bash.inc
    echo ""
fi

# Check if authenticated
echo "Step 1: Authentication Check"
echo "─────────────────────────────────────────────────────────────────────────"
if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1)
    echo -e "${GREEN}✅ Already authenticated as: ${ACCOUNT}${NC}"
    echo ""
    read -p "Do you want to use this account? (Y/n): " USE_ACCOUNT
    if [ "$USE_ACCOUNT" = "n" ] || [ "$USE_ACCOUNT" = "N" ]; then
        gcloud auth login
    fi
else
    echo "Not authenticated yet. Let's login..."
    echo ""
    gcloud auth login
fi

echo ""
echo "Step 2: Project Information"
echo "─────────────────────────────────────────────────────────────────────────"

# Check current project
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)

if [ -n "$CURRENT_PROJECT" ] && [ "$CURRENT_PROJECT" != "(unset)" ]; then
    echo -e "${GREEN}✅ Current project: ${CURRENT_PROJECT}${NC}"
    echo ""
    read -p "Do you want to use this project? (Y/n): " USE_PROJECT
    if [ "$USE_PROJECT" != "n" ] && [ "$USE_PROJECT" != "N" ]; then
        PROJECT_ID="$CURRENT_PROJECT"
    fi
fi

# If no project or user wants different one
if [ -z "$PROJECT_ID" ]; then
    echo ""
    echo "Available projects in your account:"
    echo ""
    gcloud projects list --format="table(projectId,name,projectNumber)"
    echo ""
    
    read -p "Do you want to create a new project? (y/N): " CREATE_NEW
    
    if [ "$CREATE_NEW" = "y" ] || [ "$CREATE_NEW" = "Y" ]; then
        echo ""
        read -p "Enter new project name (e.g., ai-sec-training): " PROJECT_NAME
        
        # Generate project ID
        RANDOM_SUFFIX=$(date +%s | tail -c 7)
        PROJECT_ID="${PROJECT_NAME}-${RANDOM_SUFFIX}"
        
        echo ""
        echo "Creating project:"
        echo "  Name: ${PROJECT_NAME}"
        echo "  ID: ${PROJECT_ID}"
        echo ""
        
        gcloud projects create ${PROJECT_ID} --name="${PROJECT_NAME}"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Project created successfully!${NC}"
            gcloud config set project ${PROJECT_ID}
        else
            echo "Failed to create project. Please try a different name."
            exit 1
        fi
    else
        echo ""
        read -p "Enter the Project ID you want to use: " PROJECT_ID
        gcloud config set project ${PROJECT_ID}
    fi
fi

# Verify project
echo ""
echo "Verifying project..."
if gcloud projects describe ${PROJECT_ID} &>/dev/null; then
    PROJECT_NAME=$(gcloud projects describe ${PROJECT_ID} --format="value(name)")
    PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
    echo -e "${GREEN}✅ Project verified!${NC}"
else
    echo -e "${YELLOW}⚠️  Could not verify project. Please check the ID.${NC}"
    exit 1
fi

echo ""
echo "Step 3: Bucket Information"
echo "─────────────────────────────────────────────────────────────────────────"

BUCKET_NAME="${PROJECT_ID}-ml-data"

echo "Your bucket name will be: ${BUCKET_NAME}"
echo ""
echo "Checking if bucket exists..."

if gsutil ls gs://${BUCKET_NAME} &>/dev/null; then
    echo -e "${GREEN}✅ Bucket already exists: gs://${BUCKET_NAME}${NC}"
else
    echo "Bucket doesn't exist yet."
    echo "The setup script will create it automatically."
fi

echo ""
echo "================================================================================================"
echo -e "${GREEN}✅ SUMMARY${NC}"
echo "================================================================================================"
echo ""
echo "📋 Your GCP Information:"
echo ""
echo "  🔹 Project Name    : ${PROJECT_NAME}"
echo "  🔹 Project ID      : ${PROJECT_ID}"
echo "  🔹 Project Number  : ${PROJECT_NUMBER}"
echo "  🔹 Bucket Name     : ${BUCKET_NAME}"
echo "  🔹 Bucket Path     : gs://${BUCKET_NAME}"
echo ""
echo "================================================================================================"
echo -e "${BLUE}📝 NEXT STEPS${NC}"
echo "================================================================================================"
echo ""
echo "1. Run the setup script:"
echo "   ${GREEN}./setup_gcp_vertex.sh${NC}"
echo "   (Just press Enter when asked for Project ID to use: ${PROJECT_ID})"
echo ""
echo "2. Request GPU quota (follow instructions in setup script)"
echo ""
echo "3. Submit training job:"
echo "   ${GREEN}./submit_vertex_training.sh ${PROJECT_ID}${NC}"
echo ""
echo "================================================================================================"
echo ""

# Save to file for reference
cat > .gcp_config << EOF
# GCP Configuration for AI Security Training
# Generated on $(date)

PROJECT_ID="${PROJECT_ID}"
PROJECT_NAME="${PROJECT_NAME}"
PROJECT_NUMBER="${PROJECT_NUMBER}"
BUCKET_NAME="${BUCKET_NAME}"
REGION="us-central1"

# Quick commands:
# Set project: gcloud config set project ${PROJECT_ID}
# List buckets: gsutil ls
# Training: ./submit_vertex_training.sh ${PROJECT_ID}
EOF

echo -e "${GREEN}✅ Configuration saved to: .gcp_config${NC}"
echo ""
echo "You can source this file in the future:"
echo "  source .gcp_config"
echo ""

# Export for current session
export PROJECT_ID
export BUCKET_NAME
export REGION="us-central1"

echo -e "${GREEN}✅ Environment variables set for this session:${NC}"
echo "  \$PROJECT_ID  = ${PROJECT_ID}"
echo "  \$BUCKET_NAME = ${BUCKET_NAME}"
echo "  \$REGION      = ${REGION}"
echo ""
echo "================================================================================================"
