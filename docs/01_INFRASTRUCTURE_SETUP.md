# Infrastructure Setup Guide

## Overview
This guide walks through setting up the complete infrastructure for the AI AppSec Platform, from development environment to production deployment.

---

## Table of Contents
1. [Development Environment](#development-environment)
2. [Kubernetes Cluster Setup](#kubernetes-cluster-setup)
3. [Core Services Deployment](#core-services-deployment)
4. [Security Tools Installation](#security-tools-installation)
5. [GPU Configuration](#gpu-configuration)
6. [Monitoring & Logging](#monitoring--logging)

---

## 1. Development Environment

### 1.1 Hardware Requirements

**Minimum Development Setup:**
```
CPU: 16 cores (Intel/AMD)
RAM: 64GB
GPU: NVIDIA RTX 3090/4090 (24GB VRAM) or better
Storage: 1TB NVMe SSD
OS: Ubuntu 22.04 LTS
```

**Recommended Training Setup:**
```
CPU: 32+ cores (AMD EPYC or Intel Xeon)
RAM: 128-256GB
GPU: NVIDIA A100 (40GB/80GB) or 2x RTX 4090
Storage: 2TB NVMe SSD (PCIe 4.0)
Network: 10Gbps
```

### 1.2 Software Prerequisites

#### Install NVIDIA Drivers and CUDA

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA driver (version 535+)
sudo apt install -y nvidia-driver-535

# Reboot
sudo reboot

# Verify driver installation
nvidia-smi

# Install CUDA Toolkit 12.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA
nvcc --version
```

#### Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version

# Install Docker Compose
sudo apt install -y docker-compose-plugin

# Verify
docker compose version
```

#### Install NVIDIA Container Toolkit

```bash
# Setup package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt update
sudo apt install -y nvidia-docker2

# Restart Docker daemon
sudo systemctl restart docker

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 1.3 Python Environment Setup

```bash
# Install Python 3.10+
sudo apt install -y python3.10 python3.10-venv python3-pip

# Create virtual environment
python3.10 -m venv ~/ai-appsec-env
source ~/ai-appsec-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Install core ML libraries
pip install \
  transformers>=4.36.0 \
  accelerate>=0.25.0 \
  peft>=0.7.0 \
  bitsandbytes>=0.41.0 \
  scipy \
  sentencepiece \
  protobuf

# Install training frameworks
pip install \
  axolotl \
  deepspeed \
  flash-attn --no-build-isolation

# Install MLOps tools
pip install \
  mlflow \
  wandb \
  ray[train]

# Install API and database tools
pip install \
  fastapi \
  uvicorn[standard] \
  sqlalchemy \
  psycopg2-binary \
  redis \
  celery \
  pydantic \
  python-multipart

# Install testing and utilities
pip install \
  pytest \
  pytest-asyncio \
  black \
  ruff \
  mypy
```

### 1.4 Development Tools

```bash
# Install essential tools
sudo apt install -y \
  git \
  curl \
  wget \
  vim \
  htop \
  tmux \
  tree \
  jq \
  unzip

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
kubectl version --client

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm version

# Install k9s (Kubernetes CLI)
wget https://github.com/derailed/k9s/releases/download/v0.31.7/k9s_Linux_amd64.tar.gz
tar -xzf k9s_Linux_amd64.tar.gz
sudo mv k9s /usr/local/bin/
```

---

## 2. Kubernetes Cluster Setup

### 2.1 Cloud-Based Kubernetes (Recommended for Production)

#### AWS EKS

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Configure AWS CLI
aws configure

# Create EKS cluster with GPU nodes
eksctl create cluster \
  --name ai-appsec-prod \
  --region us-west-2 \
  --version 1.28 \
  --nodegroup-name gpu-workers \
  --node-type g4dn.xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed \
  --with-oidc \
  --ssh-access \
  --ssh-public-key ~/.ssh/id_rsa.pub

# Add CPU node group
eksctl create nodegroup \
  --cluster ai-appsec-prod \
  --region us-west-2 \
  --name cpu-workers \
  --node-type m5.2xlarge \
  --nodes 5 \
  --nodes-min 2 \
  --nodes-max 20 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name ai-appsec-prod

# Verify cluster
kubectl get nodes
kubectl get nodes -L node.kubernetes.io/instance-type
```

#### GCP GKE

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Create GKE cluster
gcloud container clusters create ai-appsec-prod \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-8 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 20 \
  --enable-stackdriver-kubernetes \
  --enable-ip-alias \
  --network "default" \
  --subnetwork "default"

# Add GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster ai-appsec-prod \
  --zone us-central1-a \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --machine-type n1-standard-4 \
  --num-nodes 2 \
  --min-nodes 1 \
  --max-nodes 10 \
  --enable-autoscaling

# Get credentials
gcloud container clusters get-credentials ai-appsec-prod --zone us-central1-a

# Install NVIDIA GPU device plugin
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
```

### 2.2 On-Premise Kubernetes

#### Using kubeadm

```bash
# On all nodes: Install container runtime (containerd)
sudo apt update
sudo apt install -y containerd

# Configure containerd
sudo mkdir -p /etc/containerd
containerd config default | sudo tee /etc/containerd/config.toml

# Enable SystemdCgroup
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml

# Restart containerd
sudo systemctl restart containerd
sudo systemctl enable containerd

# Disable swap
sudo swapoff -a
sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab

# Install kubeadm, kubelet, kubectl
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-archive-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt update
sudo apt install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

# On master node: Initialize cluster
sudo kubeadm init --pod-network-cidr=10.244.0.0/16

# Set up kubectl for current user
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Install CNI plugin (Calico)
kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml

# On worker nodes: Join the cluster (use the command from kubeadm init output)
sudo kubeadm join <master-ip>:6443 --token <token> --discovery-token-ca-cert-hash sha256:<hash>

# Verify cluster
kubectl get nodes
```

#### Install NVIDIA GPU Operator (for GPU nodes)

```bash
# Add NVIDIA Helm repository
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install GPU Operator
helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator

# Verify GPU resources
kubectl get nodes -o=custom-columns=NAME:.metadata.name,GPUs:.status.capacity.'nvidia\.com/gpu'
```

### 2.3 Storage Configuration

#### Install NFS Provisioner (for on-prem)

```bash
# Install NFS server (on a dedicated storage node)
sudo apt install -y nfs-kernel-server

# Create NFS share directory
sudo mkdir -p /mnt/nfs_share
sudo chown nobody:nogroup /mnt/nfs_share
sudo chmod 777 /mnt/nfs_share

# Configure exports
echo "/mnt/nfs_share *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -a
sudo systemctl restart nfs-kernel-server

# On Kubernetes: Install NFS CSI driver
helm repo add csi-driver-nfs https://raw.githubusercontent.com/kubernetes-csi/csi-driver-nfs/master/charts
helm install csi-driver-nfs csi-driver-nfs/csi-driver-nfs \
  --namespace kube-system \
  --set kubeletDir=/var/lib/kubelet

# Create StorageClass
cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: nfs-csi
provisioner: nfs.csi.k8s.io
parameters:
  server: <NFS_SERVER_IP>
  share: /mnt/nfs_share
reclaimPolicy: Retain
volumeBindingMode: Immediate
EOF
```

#### Cloud Storage (AWS EBS/GCP PD)

```bash
# AWS: EBS CSI driver is usually pre-installed on EKS
kubectl get storageclass

# If not, install:
kubectl apply -k "github.com/kubernetes-sigs/aws-ebs-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.25"

# GCP: PD CSI driver is pre-installed on GKE
kubectl get storageclass
```

---

## 3. Core Services Deployment

### 3.1 Namespace Setup

```bash
# Create namespaces
kubectl create namespace ai-appsec
kubectl create namespace monitoring
kubectl create namespace security-tools

# Set default namespace
kubectl config set-context --current --namespace=ai-appsec
```

### 3.2 PostgreSQL Database

```bash
# Add Bitnami Helm repo
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Install PostgreSQL
helm install postgresql bitnami/postgresql \
  --namespace ai-appsec \
  --set auth.username=appsec \
  --set auth.password=SecurePassword123! \
  --set auth.database=appsec_db \
  --set primary.persistence.size=100Gi \
  --set primary.resources.requests.memory=4Gi \
  --set primary.resources.requests.cpu=2 \
  --set primary.resources.limits.memory=8Gi \
  --set primary.resources.limits.cpu=4

# Get connection details
export POSTGRES_PASSWORD=$(kubectl get secret --namespace ai-appsec postgresql -o jsonpath="{.data.password}" | base64 -d)
echo "PostgreSQL password: $POSTGRES_PASSWORD"

# Test connection
kubectl run postgresql-client --rm --tty -i --restart='Never' \
  --namespace ai-appsec \
  --image docker.io/bitnami/postgresql:16 \
  --env="PGPASSWORD=$POSTGRES_PASSWORD" \
  --command -- psql --host postgresql -U appsec -d appsec_db -p 5432
```

### 3.3 Redis Cache

```bash
# Install Redis
helm install redis bitnami/redis \
  --namespace ai-appsec \
  --set auth.password=RedisPassword123! \
  --set master.persistence.size=20Gi \
  --set replica.replicaCount=2 \
  --set replica.persistence.size=20Gi

# Get Redis password
export REDIS_PASSWORD=$(kubectl get secret --namespace ai-appsec redis -o jsonpath="{.data.redis-password}" | base64 -d)
echo "Redis password: $REDIS_PASSWORD"

# Test connection
kubectl run --namespace ai-appsec redis-client --rm --tty -i --restart='Never' \
  --env REDIS_PASSWORD=$REDIS_PASSWORD \
  --image docker.io/bitnami/redis:7.2 -- bash

# Inside the pod:
redis-cli -h redis-master -a $REDIS_PASSWORD
# Test: SET test "Hello"
# Test: GET test
```

### 3.4 MinIO Object Storage

```bash
# Install MinIO
helm install minio bitnami/minio \
  --namespace ai-appsec \
  --set auth.rootUser=admin \
  --set auth.rootPassword=MinioPassword123! \
  --set defaultBuckets=models,datasets,artifacts \
  --set persistence.size=500Gi \
  --set resources.requests.memory=2Gi \
  --set resources.requests.cpu=1

# Get MinIO credentials
export MINIO_ROOT_USER=$(kubectl get secret --namespace ai-appsec minio -o jsonpath="{.data.root-user}" | base64 -d)
export MINIO_ROOT_PASSWORD=$(kubectl get secret --namespace ai-appsec minio -o jsonpath="{.data.root-password}" | base64 -d)

echo "MinIO User: $MINIO_ROOT_USER"
echo "MinIO Password: $MINIO_ROOT_PASSWORD"

# Port-forward to access MinIO UI
kubectl port-forward --namespace ai-appsec svc/minio 9001:9001

# Open browser: http://localhost:9001
```

### 3.5 RabbitMQ Message Queue

```bash
# Install RabbitMQ
helm install rabbitmq bitnami/rabbitmq \
  --namespace ai-appsec \
  --set auth.username=admin \
  --set auth.password=RabbitPassword123! \
  --set persistence.size=20Gi \
  --set resources.requests.memory=2Gi \
  --set resources.requests.cpu=1

# Get RabbitMQ password
export RABBITMQ_PASSWORD=$(kubectl get secret --namespace ai-appsec rabbitmq -o jsonpath="{.data.rabbitmq-password}" | base64 -d)
echo "RabbitMQ password: $RABBITMQ_PASSWORD"

# Access RabbitMQ management UI
kubectl port-forward --namespace ai-appsec svc/rabbitmq 15672:15672

# Open browser: http://localhost:15672
# Username: admin, Password: <RABBITMQ_PASSWORD>
```

---

## 4. Security Tools Installation

### 4.1 Semgrep (SAST)

```bash
# Install Semgrep CLI
pip install semgrep

# Verify installation
semgrep --version

# Download community rules
semgrep --config=auto /path/to/test/code

# For Kubernetes deployment, create a Docker image:
cat <<EOF > Dockerfile.semgrep
FROM python:3.10-slim
RUN pip install semgrep
ENTRYPOINT ["semgrep"]
EOF

docker build -t ai-appsec/semgrep:latest -f Dockerfile.semgrep .
```

### 4.2 Trivy (Container & SCA)

```bash
# Install Trivy CLI
sudo apt install -y wget apt-transport-https gnupg lsb-release
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | gpg --dearmor | sudo tee /usr/share/keyrings/trivy.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/trivy.gpg] https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
sudo apt update
sudo apt install -y trivy

# Verify installation
trivy --version

# Update vulnerability database
trivy image --download-db-only

# Test scan
trivy image python:3.10-slim
```

### 4.3 OWASP ZAP (DAST)

```bash
# Pull ZAP Docker image
docker pull zaproxy/zap-stable

# Test ZAP
docker run --rm -v $(pwd):/zap/wrk:rw zaproxy/zap-stable zap-baseline.py -t https://example.com

# For Kubernetes deployment
kubectl create deployment zap \
  --image=zaproxy/zap-stable \
  --namespace=security-tools
```

### 4.4 OSV-Scanner (SCA)

```bash
# Install OSV-Scanner
go install github.com/google/osv-scanner/cmd/osv-scanner@latest

# Or download binary
wget https://github.com/google/osv-scanner/releases/latest/download/osv-scanner_linux_amd64
chmod +x osv-scanner_linux_amd64
sudo mv osv-scanner_linux_amd64 /usr/local/bin/osv-scanner

# Verify
osv-scanner --version

# Test
osv-scanner -r /path/to/project
```

### 4.5 Checkov (IaC Security)

```bash
# Install Checkov
pip install checkov

# Verify
checkov --version

# Test
checkov -d /path/to/terraform/code
```

### 4.6 Syft (SBOM Generation)

```bash
# Install Syft
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin

# Verify
syft version

# Test - generate SBOM
syft packages python:3.10-slim -o cyclonedx-json > sbom.json
```

---

## 5. GPU Configuration

### 5.1 NVIDIA Device Plugin

```bash
# Deploy NVIDIA device plugin (if not using GPU Operator)
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPU nodes
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"
```

### 5.2 GPU Resource Limits

```yaml
# Example pod with GPU request
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
  namespace: ai-appsec
spec:
  containers:
  - name: cuda-container
    image: nvidia/cuda:12.1.0-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
  restartPolicy: Never
EOF

# Check logs
kubectl logs gpu-test -n ai-appsec
```

### 5.3 GPU Time-Slicing (Optional - for development)

```yaml
# ConfigMap for GPU time-slicing (share GPU among multiple pods)
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  any: |-
    version: v1
    sharing:
      timeSlicing:
        replicas: 4
EOF

# Patch GPU Operator
kubectl patch clusterpolicy/cluster-policy \
  -n gpu-operator --type merge \
  -p '{"spec": {"devicePlugin": {"config": {"name": "time-slicing-config"}}}}'
```

---

## 6. Monitoring & Logging

### 6.1 Prometheus Stack

```bash
# Add Prometheus community Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack (Prometheus + Grafana + Alertmanager)
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi \
  --set grafana.adminPassword=GrafanaAdmin123!

# Get Grafana admin password
kubectl get secret --namespace monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 -d

# Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Open browser: http://localhost:3000
# Username: admin, Password: <from above>
```

### 6.2 ELK Stack (Elasticsearch, Logstash, Kibana)

```bash
# Add Elastic Helm repo
helm repo add elastic https://helm.elastic.co
helm repo update

# Install Elasticsearch
helm install elasticsearch elastic/elasticsearch \
  --namespace monitoring \
  --set replicas=3 \
  --set volumeClaimTemplate.resources.requests.storage=100Gi \
  --set resources.requests.memory=4Gi \
  --set resources.limits.memory=8Gi

# Install Kibana
helm install kibana elastic/kibana \
  --namespace monitoring \
  --set service.type=LoadBalancer

# Install Filebeat (log collector)
helm install filebeat elastic/filebeat \
  --namespace monitoring

# Get Kibana URL
kubectl get svc kibana-kibana -n monitoring
```

### 6.3 NVIDIA DCGM Exporter (GPU Metrics)

```bash
# Install DCGM Exporter
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm repo update

helm install dcgm-exporter gpu-helm-charts/dcgm-exporter \
  --namespace monitoring \
  --set serviceMonitor.enabled=true \
  --set serviceMonitor.namespace=monitoring

# Import DCGM dashboard in Grafana
# Dashboard ID: 12239
```

### 6.4 Custom ServiceMonitor for AI Models

```yaml
cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ai-model-metrics
  namespace: ai-appsec
spec:
  selector:
    matchLabels:
      app: ai-model-server
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
EOF
```

---

## 7. Verification Checklist

### Infrastructure Health Checks

```bash
# Check all nodes are ready
kubectl get nodes

# Check all system pods
kubectl get pods -A

# Check storage classes
kubectl get sc

# Check persistent volumes
kubectl get pv

# Check services
kubectl get svc -A

# Check GPU availability
kubectl get nodes -o=custom-columns=NAME:.metadata.name,GPUs:.status.capacity.'nvidia\.com/gpu'

# Test GPU workload
kubectl run gpu-test --rm -it --image=nvidia/cuda:12.1.0-base-ubuntu22.04 --restart=Never -- nvidia-smi

# Check monitoring
kubectl get pods -n monitoring

# Check security tools
kubectl get pods -n security-tools
```

### Database Connections

```bash
# Test PostgreSQL
kubectl run postgresql-client --rm --tty -i --restart='Never' \
  --namespace ai-appsec \
  --image docker.io/bitnami/postgresql:16 \
  --env="PGPASSWORD=$POSTGRES_PASSWORD" \
  --command -- psql --host postgresql -U appsec -d appsec_db -p 5432

# Test Redis
kubectl run --namespace ai-appsec redis-client --rm --tty -i --restart='Never' \
  --env REDIS_PASSWORD=$REDIS_PASSWORD \
  --image docker.io/bitnami/redis:7.2 -- redis-cli -h redis-master -a $REDIS_PASSWORD ping

# Test MinIO (using mc client)
kubectl run minio-client --rm -it --restart='Never' \
  --namespace ai-appsec \
  --image minio/mc \
  -- mc alias set myminio http://minio:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD
```

---

## 8. Next Steps

✅ Infrastructure is now ready for:
1. **Data pipeline deployment** (see `02_DATA_PREPARATION.md`)
2. **Model training setup** (see `03_MODEL_TRAINING.md`)
3. **Application deployment** (see `05_MODEL_DEPLOYMENT.md`)

---

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check NVIDIA runtime in Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check GPU operator logs
kubectl logs -n gpu-operator -l app=nvidia-device-plugin-daemonset

# Recreate GPU plugin
kubectl delete pod -n gpu-operator -l app=nvidia-device-plugin-daemonset
```

### Pod Stuck in Pending

```bash
# Describe pod to see events
kubectl describe pod <pod-name> -n <namespace>

# Common issues:
# - Insufficient resources: Scale cluster or reduce requests
# - PVC not bound: Check storage class and PV availability
# - Image pull errors: Check registry credentials
```

### High Memory Usage

```bash
# Check node memory
kubectl top nodes

# Check pod memory
kubectl top pods -A

# Identify memory hogs
kubectl get pods -A --sort-by='.status.containerStatuses[0].restartCount'
```

---

## Cost Optimization Tips

1. **Use Spot/Preemptible Instances** for non-critical workloads
2. **Auto-scaling**: Configure HPA and Cluster Autoscaler
3. **Resource Requests/Limits**: Set appropriate values to avoid over-provisioning
4. **Storage**: Use lifecycle policies to delete old data
5. **GPU Sharing**: Use time-slicing for development environments

---

## Security Hardening

```bash
# Enable Pod Security Standards
kubectl label namespace ai-appsec pod-security.kubernetes.io/enforce=baseline

# Create NetworkPolicies
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: ai-appsec
spec:
  podSelector: {}
  policyTypes:
  - Ingress
EOF

# Enable audit logging (cloud providers have specific methods)

# Rotate secrets regularly
kubectl create secret generic db-secret \
  --from-literal=password=$(openssl rand -base64 32) \
  --dry-run=client -o yaml | kubectl apply -f -
```
