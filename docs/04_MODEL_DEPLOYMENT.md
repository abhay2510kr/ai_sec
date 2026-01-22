# Model Deployment and Serving Guide

## Overview
This guide covers deploying fine-tuned CodeLlama 7B models for production inference, including model serving setup, API endpoints, autoscaling, and monitoring.

---

## Table of Contents
1. [Model Serving Options](#model-serving-options)
2. [vLLM Deployment](#vllm-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [API Gateway Setup](#api-gateway-setup)
5. [Load Balancing and Autoscaling](#load-balancing-and-autoscaling)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Performance Optimization](#performance-optimization)

---

## 1. Model Serving Options

### Comparison of Serving Frameworks

| Framework | Throughput | Latency | GPU Efficiency | Features |
|-----------|-----------|---------|----------------|----------|
| **vLLM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | PagedAttention, continuous batching |
| **TGI** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Streaming, quantization |
| **Ollama** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Easy local deployment |
| **FastAPI + Transformers** | ⭐⭐ | ⭐⭐ | ⭐⭐ | Simple, flexible |

**Recommendation: vLLM** for production (best performance/efficiency)

---

## 2. vLLM Deployment

### 2.1 Install vLLM

```bash
# Install vLLM
pip install vllm

# Verify installation
python -c "import vllm; print(vllm.__version__)"
```

### 2.2 Convert Models to vLLM Format

```python
# File: scripts/prepare_models_for_vllm.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def merge_and_save_for_vllm(base_model_path, adapter_path, output_path):
    """Merge LoRA adapter and save for vLLM"""
    
    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging adapter weights...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print("✅ Model ready for vLLM deployment")

# Prepare all models
models = {
    'sast': ('~/models/base/CodeLlama-7b-Python-hf', '~/models/codellama-7b-sast-v1'),
    'sca': ('~/models/base/CodeLlama-7b-hf', '~/models/codellama-7b-sca-v1'),
    'iac': ('~/models/base/CodeLlama-7b-hf', '~/models/codellama-7b-iac-v1'),
    'container': ('~/models/base/CodeLlama-7b-hf', '~/models/codellama-7b-container-v1'),
    'api': ('~/models/base/CodeLlama-7b-Instruct-hf', '~/models/codellama-7b-api-v1'),
    'dast': ('~/models/base/CodeLlama-7b-Instruct-hf', '~/models/codellama-7b-dast-v1'),
}

for domain, (base_path, adapter_path) in models.items():
    output_path = f'~/models/codellama-7b-{domain}-v1-vllm'
    merge_and_save_for_vllm(base_path, adapter_path, output_path)
```

### 2.3 Launch vLLM Servers

```bash
# SAST Model Server
vllm serve ~/models/codellama-7b-sast-v1-vllm \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 2048 \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
  --trust-remote-code &

# SCA Model Server
vllm serve ~/models/codellama-7b-sca-v1-vllm \
  --host 0.0.0.0 \
  --port 8002 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 1024 \
  --dtype float16 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64 &

# IaC Model Server
vllm serve ~/models/codellama-7b-iac-v1-vllm \
  --host 0.0.0.0 \
  --port 8003 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 2048 \
  --dtype float16 &

# Container Model Server
vllm serve ~/models/codellama-7b-container-v1-vllm \
  --host 0.0.0.0 \
  --port 8004 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 1024 \
  --dtype float16 &

# API Model Server
vllm serve ~/models/codellama-7b-api-v1-vllm \
  --host 0.0.0.0 \
  --port 8005 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 2048 \
  --dtype float16 &

# DAST Model Server
vllm serve ~/models/codellama-7b-dast-v1-vllm \
  --host 0.0.0.0 \
  --port 8006 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 1024 \
  --dtype float16 &
```

### 2.4 Test Model Servers

```python
# File: scripts/test_vllm_servers.py

import requests
import json

def test_model_server(port, model_name, sample_prompt):
    """Test vLLM model server"""
    
    url = f"http://localhost:{port}/v1/completions"
    
    payload = {
        "model": model_name,
        "prompt": sample_prompt,
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 0.95
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ {model_name} server (port {port}): OK")
        print(f"   Response: {result['choices'][0]['text'][:100]}...")
        return True
    else:
        print(f"❌ {model_name} server (port {port}): FAILED")
        print(f"   Error: {response.text}")
        return False

# Test all servers
test_cases = [
    (8001, "sast", "<s>[INST] Analyze this Python code for SQL injection: def query(user_input): cursor.execute('SELECT * FROM users WHERE name=' + user_input) [/INST]"),
    (8002, "sca", "<s>[INST] Check this package.json for vulnerabilities: {\"dependencies\": {\"lodash\": \"4.17.15\"}} [/INST]"),
    (8003, "iac", "<s>[INST] Check this Terraform for misconfigurations: resource \"aws_s3_bucket\" \"public\" { acl = \"public-read\" } [/INST]"),
    (8004, "container", "<s>[INST] Analyze this Dockerfile: FROM ubuntu:18.04\\nRUN apt-get update [/INST]"),
    (8005, "api", "<s>[INST] Check this API endpoint for security issues: GET /api/users/{id} without authentication [/INST]"),
    (8006, "dast", "<s>[INST] Generate SQL injection payload for login form [/INST]"),
]

for port, name, prompt in test_cases:
    test_model_server(port, name, prompt)
    print()
```

### 2.5 vLLM Configuration Optimization

```python
# File: configs/vllm_config.yaml

# SAST Model Config (High Accuracy)
sast:
  port: 8001
  model_path: ~/models/codellama-7b-sast-v1-vllm
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
  max_model_len: 2048
  max_num_batched_tokens: 4096
  max_num_seqs: 32
  dtype: float16
  quantization: awq  # Optional: for 4-bit quantized models
  enable_chunked_prefill: true

# SCA Model Config (High Throughput)
sca:
  port: 8002
  model_path: ~/models/codellama-7b-sca-v1-vllm
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
  max_model_len: 1024
  max_num_batched_tokens: 8192
  max_num_seqs: 64
  dtype: float16
  enable_chunked_prefill: true

# Multi-GPU Config (for scaling)
sast_multi_gpu:
  port: 8001
  model_path: ~/models/codellama-7b-sast-v1-vllm
  tensor_parallel_size: 2  # Spread across 2 GPUs
  pipeline_parallel_size: 1
  gpu_memory_utilization: 0.95
```

---

## 3. Kubernetes Deployment

### 3.1 Create Docker Images

```dockerfile
# File: docker/Dockerfile.vllm-sast

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM
RUN pip install --no-cache-dir \
    vllm==0.3.0 \
    fastapi \
    uvicorn

# Copy model
COPY models/codellama-7b-sast-v1-vllm /models/sast

# Expose port
EXPOSE 8001

# Run vLLM server
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/models/sast", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--gpu-memory-utilization", "0.9", \
     "--max-model-len", "2048", \
     "--dtype", "float16"]
```

```bash
# Build Docker images for all models
docker build -f docker/Dockerfile.vllm-sast -t ai-appsec/sast-model:v1 .
docker build -f docker/Dockerfile.vllm-sca -t ai-appsec/sca-model:v1 .
docker build -f docker/Dockerfile.vllm-iac -t ai-appsec/iac-model:v1 .
docker build -f docker/Dockerfile.vllm-container -t ai-appsec/container-model:v1 .
docker build -f docker/Dockerfile.vllm-api -t ai-appsec/api-model:v1 .
docker build -f docker/Dockerfile.vllm-dast -t ai-appsec/dast-model:v1 .

# Push to container registry
docker tag ai-appsec/sast-model:v1 <your-registry>/ai-appsec/sast-model:v1
docker push <your-registry>/ai-appsec/sast-model:v1
# Repeat for all models
```

### 3.2 Kubernetes Deployment Manifests

```yaml
# File: k8s/sast-model-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: sast-model
  namespace: ai-appsec
  labels:
    app: sast-model
    component: inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sast-model
  template:
    metadata:
      labels:
        app: sast-model
        component: inference
    spec:
      nodeSelector:
        node-type: gpu
      containers:
      - name: vllm
        image: <your-registry>/ai-appsec/sast-model:v1
        ports:
        - containerPort: 8001
          name: http
        - containerPort: 9090
          name: metrics
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
            cpu: 4
          limits:
            nvidia.com/gpu: 1
            memory: 24Gi
            cpu: 8
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: sast-model
  namespace: ai-appsec
  labels:
    app: sast-model
spec:
  selector:
    app: sast-model
  ports:
  - name: http
    port: 8001
    targetPort: 8001
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
```

```bash
# Deploy all model services
kubectl apply -f k8s/sast-model-deployment.yaml
kubectl apply -f k8s/sca-model-deployment.yaml
kubectl apply -f k8s/iac-model-deployment.yaml
kubectl apply -f k8s/container-model-deployment.yaml
kubectl apply -f k8s/api-model-deployment.yaml
kubectl apply -f k8s/dast-model-deployment.yaml

# Verify deployments
kubectl get pods -n ai-appsec
kubectl get svc -n ai-appsec
```

### 3.3 ConfigMap for Model Configuration

```yaml
# File: k8s/model-config.yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: ai-appsec
data:
  models.json: |
    {
      "sast": {
        "name": "SAST Vulnerability Detection",
        "endpoint": "http://sast-model:8001",
        "max_tokens": 512,
        "temperature": 0.1,
        "timeout": 30
      },
      "sca": {
        "name": "Software Composition Analysis",
        "endpoint": "http://sca-model:8002",
        "max_tokens": 512,
        "temperature": 0.1,
        "timeout": 30
      },
      "iac": {
        "name": "IaC Security Scanner",
        "endpoint": "http://iac-model:8003",
        "max_tokens": 512,
        "temperature": 0.1,
        "timeout": 30
      },
      "container": {
        "name": "Container Security Scanner",
        "endpoint": "http://container-model:8004",
        "max_tokens": 512,
        "temperature": 0.1,
        "timeout": 30
      },
      "api": {
        "name": "API Security Tester",
        "endpoint": "http://api-model:8005",
        "max_tokens": 512,
        "temperature": 0.1,
        "timeout": 30
      },
      "dast": {
        "name": "Dynamic Application Security Testing",
        "endpoint": "http://dast-model:8006",
        "max_tokens": 512,
        "temperature": 0.1,
        "timeout": 30
      }
    }
```

---

## 4. API Gateway Setup

### 4.1 FastAPI Gateway Service

```python
# File: services/api_gateway/main.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import httpx
import json
import logging
from prometheus_client import Counter, Histogram, generate_latest
import time

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['model', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration', ['model'])

app = FastAPI(title="AI AppSec API Gateway", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
with open('/config/models.json') as f:
    MODEL_CONFIG = json.load(f)

# Request/Response models
class AnalysisRequest(BaseModel):
    code: str
    language: Optional[str] = "python"
    context: Optional[Dict] = {}
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1

class AnalysisResponse(BaseModel):
    vulnerabilities: List[Dict]
    summary: str
    confidence: float
    execution_time: float

class SCARequest(BaseModel):
    manifest: str
    manifest_type: str  # package.json, requirements.txt, etc.
    ecosystem: Optional[str] = None

class IaCRequest(BaseModel):
    code: str
    iac_type: str  # terraform, cloudformation, kubernetes

# Helper function to call model
async def call_model(model_name: str, prompt: str, max_tokens: int = 512, temperature: float = 0.1):
    """Call vLLM model endpoint"""
    
    config = MODEL_CONFIG.get(model_name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    endpoint = f"{config['endpoint']}/v1/completions"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95
    }
    
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=config.get('timeout', 30)) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            
            duration = time.time() - start_time
            REQUEST_DURATION.labels(model=model_name).observe(duration)
            REQUEST_COUNT.labels(model=model_name, status='success').inc()
            
            return response.json()['choices'][0]['text'], duration
    
    except Exception as e:
        REQUEST_COUNT.labels(model=model_name, status='error').inc()
        logger.error(f"Error calling {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

# Endpoints
@app.post("/api/v1/scan/sast", response_model=AnalysisResponse)
async def scan_sast(request: AnalysisRequest):
    """SAST vulnerability detection"""
    
    prompt = f"""<s>[INST] You are a security code analyzer. Analyze the following code for vulnerabilities.

Language: {request.language}
Code:
```{request.language}
{request.code}
```

Identify any security vulnerabilities, provide the CWE type, severity, and explanation. [/INST]"""
    
    response_text, duration = await call_model('sast', prompt, request.max_tokens, request.temperature)
    
    # Parse response (simplified)
    vulnerabilities = []
    if "vulnerability" in response_text.lower():
        vulnerabilities.append({
            "type": "Potential Vulnerability",
            "description": response_text,
            "severity": "MEDIUM",
            "confidence": 0.8
        })
    
    return AnalysisResponse(
        vulnerabilities=vulnerabilities,
        summary=response_text[:200],
        confidence=0.85,
        execution_time=duration
    )

@app.post("/api/v1/scan/sca")
async def scan_sca(request: SCARequest):
    """Software Composition Analysis"""
    
    prompt = f"""<s>[INST] Analyze the following dependency manifest for vulnerable packages.

Manifest Type: {request.manifest_type}
Manifest:
```
{request.manifest}
```

Identify vulnerable dependencies, provide CVE IDs, severity scores, and remediation advice. [/INST]"""
    
    response_text, duration = await call_model('sca', prompt)
    
    return {
        "vulnerable_packages": [],
        "analysis": response_text,
        "execution_time": duration
    }

@app.post("/api/v1/scan/iac")
async def scan_iac(request: IaCRequest):
    """Infrastructure as Code security scan"""
    
    prompt = f"""<s>[INST] Check the following {request.iac_type} code for security misconfigurations.

Code:
```
{request.code}
```

Identify misconfigurations, policy violations, and provide remediation advice. [/INST]"""
    
    response_text, duration = await call_model('iac', prompt)
    
    return {
        "misconfigurations": [],
        "analysis": response_text,
        "execution_time": duration
    }

@app.post("/api/v1/scan/container")
async def scan_container(request: dict):
    """Container security scan"""
    
    dockerfile = request.get('dockerfile', '')
    
    prompt = f"""<s>[INST] Analyze the following Dockerfile for security issues.

Dockerfile:
```
{dockerfile}
```

Identify security best practice violations and provide recommendations. [/INST]"""
    
    response_text, duration = await call_model('container', prompt)
    
    return {
        "issues": [],
        "analysis": response_text,
        "execution_time": duration
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models": list(MODEL_CONFIG.keys())}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 4.2 Deploy API Gateway

```yaml
# File: k8s/api-gateway-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: ai-appsec
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: gateway
        image: <your-registry>/ai-appsec/api-gateway:v1
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: config
          mountPath: /config
        env:
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: 2Gi
            cpu: 1
          limits:
            memory: 4Gi
            cpu: 2
      volumes:
      - name: config
        configMap:
          name: model-config
---
apiVersion: v1
kind: Service
metadata:
  name: api-gateway
  namespace: ai-appsec
spec:
  selector:
    app: api-gateway
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 5. Load Balancing and Autoscaling

### 5.1 Horizontal Pod Autoscaler

```yaml
# File: k8s/hpa-sast-model.yaml

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sast-model-hpa
  namespace: ai-appsec
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sast-model
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: vllm_requests_per_second
      target:
        type: AverageValue
        averageValue: "50"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
```

### 5.2 Cluster Autoscaler Configuration

```yaml
# For AWS EKS
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-priority-expander
  namespace: kube-system
data:
  priorities: |-
    10:
      - .*-gpu-.*
    50:
      - .*-cpu-.*
```

---

## 6. Monitoring and Observability

### 6.1 Prometheus ServiceMonitor

```yaml
# File: k8s/servicemonitor-models.yaml

apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ai-models
  namespace: ai-appsec
spec:
  selector:
    matchLabels:
      component: inference
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

### 6.2 Grafana Dashboard

```json
{
  "dashboard": {
    "title": "AI AppSec Models Performance",
    "panels": [
      {
        "title": "Model Request Rate",
        "targets": [
          {
            "expr": "rate(api_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Model Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "DCGM_FI_DEV_GPU_UTIL"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE * 100"
          }
        ]
      }
    ]
  }
}
```

---

## 7. Performance Optimization

### 7.1 Model Caching

```python
# File: services/api_gateway/cache.py

import redis
import hashlib
import json

class ResultCache:
    def __init__(self, redis_host='redis', redis_port=6379, ttl=3600):
        self.redis = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.ttl = ttl
    
    def get_cache_key(self, model: str, prompt: str) -> str:
        """Generate cache key from model and prompt"""
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, model: str, prompt: str):
        """Get cached result"""
        key = self.get_cache_key(model, prompt)
        cached = self.redis.get(key)
        
        if cached:
            return json.loads(cached)
        return None
    
    def set(self, model: str, prompt: str, result: dict):
        """Cache result"""
        key = self.get_cache_key(model, prompt)
        self.redis.setex(key, self.ttl, json.dumps(result))
```

### 7.2 Request Batching

```python
# File: services/api_gateway/batcher.py

import asyncio
from collections import defaultdict
from typing import List, Dict

class RequestBatcher:
    def __init__(self, max_batch_size=32, max_wait_ms=100):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.batches = defaultdict(list)
        self.locks = defaultdict(asyncio.Lock)
    
    async def add_request(self, model: str, request: Dict):
        """Add request to batch"""
        async with self.locks[model]:
            self.batches[model].append(request)
            
            if len(self.batches[model]) >= self.max_batch_size:
                return await self.flush_batch(model)
            
            # Wait for more requests or timeout
            await asyncio.sleep(self.max_wait_ms / 1000)
            return await self.flush_batch(model)
    
    async def flush_batch(self, model: str) -> List:
        """Send batch to model"""
        batch = self.batches[model]
        self.batches[model] = []
        
        # Send batch to vLLM
        # vLLM automatically batches requests
        results = []
        for req in batch:
            result = await self.process_single(model, req)
            results.append(result)
        
        return results
```

---

## Summary

✅ **Model Serving**: vLLM for high-performance inference
✅ **Deployment**: Kubernetes with GPU support
✅ **API Gateway**: FastAPI with load balancing
✅ **Autoscaling**: HPA based on GPU utilization
✅ **Monitoring**: Prometheus + Grafana dashboards
✅ **Optimization**: Caching + batching for efficiency

**Expected Performance:**
- Latency: 200-500ms (p95)
- Throughput: 30-50 requests/sec per GPU
- GPU Utilization: 70-85%
- Cost: ~$2,000/month (6 models, 2 replicas each)

**Next Step**: Integration & Orchestration (see `05_INTEGRATION_ORCHESTRATION.md`)
