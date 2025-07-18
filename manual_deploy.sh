#!/bin/bash

# Manual Google Cloud Run Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Manual Google Cloud Run Deployment${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Google Cloud SDK is not installed. Please install it first:${NC}"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${YELLOW}⚠️  You are not authenticated with Google Cloud. Please run:${NC}"
    echo "gcloud auth login"
    exit 1
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ No project ID set. Please set it with:${NC}"
    echo "gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo -e "${GREEN}✅ Using project: $PROJECT_ID${NC}"

# Enable required APIs
echo -e "${BLUE}📋 Enabling required APIs...${NC}"
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Configure Docker to use gcloud as a credential helper
echo -e "${BLUE}🔐 Configuring Docker authentication...${NC}"
gcloud auth configure-docker

# Build the container image for AMD64 (Cloud Run requirement)
echo -e "${BLUE}🔨 Building Docker image for AMD64...${NC}"
docker build --platform linux/amd64 -t gcr.io/$PROJECT_ID/ball-detection-app .

# Push the container image to Container Registry
echo -e "${BLUE}📤 Pushing image to Container Registry...${NC}"
docker push gcr.io/$PROJECT_ID/ball-detection-app

# Deploy container image to Cloud Run
echo -e "${BLUE}🚀 Deploying to Cloud Run...${NC}"
gcloud run deploy ball-detection-app \
  --image gcr.io/$PROJECT_ID/ball-detection-app \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10

echo -e "${GREEN}✅ Deployment completed successfully!${NC}"

# Get the service URL
SERVICE_URL=$(gcloud run services describe ball-detection-app --region=us-central1 --format="value(status.url)")

echo -e "${GREEN}🌐 Your application is available at:${NC}"
echo -e "${BLUE}$SERVICE_URL${NC}"

echo -e "${YELLOW}📝 Note:${NC}"
echo "- The first request might take a few minutes as the container starts up"
echo "- Webcam functionality requires HTTPS and user permission"
echo "- Video upload is limited to 16MB files"
echo "- The service will scale to zero when not in use (cost optimization)"

echo -e "${GREEN}🎉 Manual deployment complete!${NC}" 