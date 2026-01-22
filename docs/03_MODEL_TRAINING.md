# Model Training Guide

## Overview
This guide provides step-by-step instructions for fine-tuning CodeLlama 7B models for each security domain using the prepared datasets.

---

## Table of Contents
1. [Training Environment Setup](#training-environment-setup)
2. [SAST Model Training](#sast-model-training)
3. [SCA Model Training](#sca-model-training)
4. [IaC Security Model Training](#iac-security-model-training)
5. [Container Security Model Training](#container-security-model-training)
6. [API Security Model Training](#api-security-model-training)
7. [DAST Model Training](#dast-model-training)
8. [Model Evaluation](#model-evaluation)
9. [Model Optimization](#model-optimization)

---

## 1. Training Environment Setup

### 1.1 Install Training Dependencies

```bash
# Activate virtual environment
source ~/ai-appsec-env/bin/activate

# Install core training libraries
pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Hugging Face ecosystem
pip install transformers==4.37.0 \
  datasets==2.16.0 \
  accelerate==0.26.0 \
  peft==0.8.0 \
  bitsandbytes==0.42.0 \
  sentencepiece \
  protobuf

# Install Axolotl (fine-tuning framework)
pip install axolotl

# Install training utilities
pip install wandb \
  mlflow \
  tensorboard \
  scipy \
  scikit-learn

# Install flash-attention for faster training
pip install flash-attn --no-build-isolation
```

### 1.2 Download CodeLlama 7B Base Model

```bash
# Create model directory
mkdir -p ~/models/base

# Download using Hugging Face CLI
huggingface-cli login  # Use your HF token

# Download CodeLlama 7B Python variant (best for SAST)
huggingface-cli download codellama/CodeLlama-7b-Python-hf \
  --local-dir ~/models/base/CodeLlama-7b-Python-hf

# Download CodeLlama 7B base (for SCA, IaC, Container)
huggingface-cli download codellama/CodeLlama-7b-hf \
  --local-dir ~/models/base/CodeLlama-7b-hf

# Download CodeLlama 7B Instruct (for DAST, API testing)
huggingface-cli download codellama/CodeLlama-7b-Instruct-hf \
  --local-dir ~/models/base/CodeLlama-7b-Instruct-hf

# Verify downloads
ls -lh ~/models/base/
```

### 1.3 MLflow Setup

```bash
# Start MLflow tracking server
mlflow server \
  --backend-store-uri sqlite:///~/mlflow/mlflow.db \
  --default-artifact-root ~/mlflow/artifacts \
  --host 0.0.0.0 \
  --port 5000 &

# Access MLflow UI: http://localhost:5000

# Set environment variable
echo "export MLFLOW_TRACKING_URI=http://localhost:5000" >> ~/.bashrc
source ~/.bashrc
```

### 1.4 Weights & Biases Setup (Optional)

```bash
# Install W&B
pip install wandb

# Login
wandb login

# Initialize project
wandb init --project ai-appsec-training
```

---

## 2. SAST Model Training

### 2.1 Training Configuration

Create training config file:

```yaml
# File: configs/sast_training.yaml

base_model: ~/models/base/CodeLlama-7b-Python-hf
model_type: LlamaForCausalLM
tokenizer_type: CodeLlamaTokenizer

# Dataset
datasets:
  - path: ~/datasets/processed/sast/sast_train.json
    type: alpaca
    
val_set_size: 0.1

# Output
output_dir: ~/models/codellama-7b-sast-v1

# LoRA Configuration
adapter: lora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Quantization (QLoRA)
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true

# Training Hyperparameters
sequence_len: 2048
micro_batch_size: 4
gradient_accumulation_steps: 8
num_epochs: 3
learning_rate: 0.0002
lr_scheduler: cosine
warmup_steps: 100

# Optimization
optimizer: paged_adamw_8bit
weight_decay: 0.01
max_grad_norm: 1.0

# Advanced Features
flash_attention: true
gradient_checkpointing: true
bf16: true
tf32: true

# Logging
logging_steps: 10
eval_steps: 100
save_steps: 500
save_total_limit: 3

# MLflow
mlflow_tracking_uri: http://localhost:5000
mlflow_experiment_name: sast-vulnerability-detection

# Weights & Biases (optional)
wandb_project: ai-appsec-training
wandb_entity: your-username
wandb_watch: gradients
wandb_log_model: true

# Evaluation
evals_per_epoch: 4
eval_sample_packing: false

# Special Tokens
special_tokens:
  pad_token: "<PAD>"
```

### 2.2 Prepare Training Script

```python
# File: scripts/train_sast_model.py

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model_and_tokenizer(model_path, quantization_config):
    """Load model with 4-bit quantization"""
    logger.info(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora(model, config):
    """Configure LoRA adapters"""
    logger.info("Setting up LoRA")
    
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=config['lora_target_modules'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def load_and_prepare_dataset(data_path, tokenizer, max_length=2048):
    """Load and tokenize dataset"""
    logger.info(f"Loading dataset from {data_path}")
    
    dataset = load_dataset('json', data_files=data_path, split='train')
    
    def tokenize_function(examples):
        # Format: instruction + response
        texts = []
        for instruction, response in zip(examples['instruction'], examples['response']):
            text = f"<s>[INST] {instruction} [/INST] {response} </s>"
            texts.append(text)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Labels are the same as input_ids for causal LM
        tokenized['labels'] = tokenized['input_ids'].clone()
        
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset

def train_model(config_path='configs/sast_training.yaml'):
    """Main training function"""
    import yaml
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup MLflow
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    mlflow.set_experiment(config['mlflow_experiment_name'])
    
    with mlflow.start_run():
        # Log config
        mlflow.log_params(config)
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # Setup model
        model, tokenizer = setup_model_and_tokenizer(
            config['base_model'],
            bnb_config
        )
        
        # Setup LoRA
        model = setup_lora(model, config)
        
        # Load dataset
        dataset = load_and_prepare_dataset(
            config['datasets'][0]['path'],
            tokenizer,
            config['sequence_len']
        )
        
        # Split train/val
        split_dataset = dataset.train_test_split(
            test_size=config['val_set_size'],
            seed=42
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=config['output_dir'],
            num_train_epochs=config['num_epochs'],
            per_device_train_batch_size=config['micro_batch_size'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            learning_rate=config['learning_rate'],
            lr_scheduler_type=config['lr_scheduler'],
            warmup_steps=config['warmup_steps'],
            logging_steps=config['logging_steps'],
            save_steps=config['save_steps'],
            eval_steps=config['eval_steps'],
            evaluation_strategy="steps",
            save_total_limit=config['save_total_limit'],
            bf16=config['bf16'],
            tf32=config['tf32'],
            optim=config['optimizer'],
            weight_decay=config['weight_decay'],
            max_grad_norm=config['max_grad_norm'],
            gradient_checkpointing=config['gradient_checkpointing'],
            report_to=["mlflow", "tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split_dataset['train'],
            eval_dataset=split_dataset['test'],
            tokenizer=tokenizer
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {config['output_dir']}")
        trainer.save_model()
        tokenizer.save_pretrained(config['output_dir'])
        
        # Log final metrics
        final_metrics = trainer.evaluate()
        mlflow.log_metrics(final_metrics)
        
        # Save model to MLflow
        mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "tokenizer": tokenizer
            },
            artifact_path="model"
        )
        
        logger.info("Training complete!")

if __name__ == "__main__":
    train_model('configs/sast_training.yaml')
```

### 2.3 Launch Training

```bash
# Single GPU training
python scripts/train_sast_model.py

# Multi-GPU training (with accelerate)
accelerate launch --config_file configs/accelerate_config.yaml \
  scripts/train_sast_model.py

# Using Axolotl (alternative)
axolotl train configs/sast_training.yaml
```

### 2.4 Monitor Training

```bash
# Monitor with TensorBoard
tensorboard --logdir ~/models/codellama-7b-sast-v1/runs

# Monitor with MLflow
# Open http://localhost:5000 in browser

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f ~/models/codellama-7b-sast-v1/training.log
```

### 2.5 Expected Training Time and Metrics

**Training Time (Single A100 80GB):**
- Dataset: 80,000 samples
- Batch size: 4, Gradient accumulation: 8 (effective batch size: 32)
- Epochs: 3
- **Estimated time: 24-30 hours**

**Expected Metrics:**
- Final training loss: 0.3-0.5
- Final validation loss: 0.4-0.6
- Perplexity: 1.5-1.8
- VRAM usage: ~40GB (with 4-bit quantization)

---

## 3. SCA Model Training

### 3.1 Training Configuration

```yaml
# File: configs/sca_training.yaml

base_model: ~/models/base/CodeLlama-7b-hf  # Base variant, not Python
model_type: LlamaForCausalLM
tokenizer_type: CodeLlamaTokenizer

datasets:
  - path: ~/datasets/processed/sca/sca_train.json
    type: alpaca

val_set_size: 0.1
output_dir: ~/models/codellama-7b-sca-v1

# LoRA Config (same as SAST)
adapter: lora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Quantization
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16

# Training params (adjusted for SCA)
sequence_len: 1024  # Shorter for dependency manifests
micro_batch_size: 8  # Can fit more with shorter sequences
gradient_accumulation_steps: 4
num_epochs: 3
learning_rate: 0.0002

# Optimization
optimizer: paged_adamw_8bit
flash_attention: true
gradient_checkpointing: true
bf16: true

# Logging
logging_steps: 10
eval_steps: 100
save_steps: 500

# MLflow
mlflow_tracking_uri: http://localhost:5000
mlflow_experiment_name: sca-vulnerability-detection
```

### 3.2 Launch SCA Training

```bash
# Create SCA-specific training script or use generic one with config
python scripts/train_model.py --config configs/sca_training.yaml

# Or with Axolotl
axolotl train configs/sca_training.yaml
```

**Expected Training Time (Single A100):**
- Dataset: 160,000 samples
- **Estimated time: 20-25 hours**

---

## 4. IaC Security Model Training

### 4.1 Training Configuration

```yaml
# File: configs/iac_training.yaml

base_model: ~/models/base/CodeLlama-7b-hf
output_dir: ~/models/codellama-7b-iac-v1

datasets:
  - path: ~/datasets/processed/iac/iac_train.json
    type: alpaca

val_set_size: 0.1

# LoRA Config
adapter: lora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Quantization
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16

# Training params
sequence_len: 2048  # IaC files can be long
micro_batch_size: 4
gradient_accumulation_steps: 8
num_epochs: 3
learning_rate: 0.0002

# Optimization
optimizer: paged_adamw_8bit
flash_attention: true
gradient_checkpointing: true
bf16: true

# MLflow
mlflow_tracking_uri: http://localhost:5000
mlflow_experiment_name: iac-security-detection
```

### 4.2 Launch IaC Training

```bash
axolotl train configs/iac_training.yaml
```

**Expected Training Time (Single A100):**
- Dataset: 40,000 samples
- **Estimated time: 12-15 hours**

---

## 5. Container Security Model Training

### 5.1 Training Configuration

```yaml
# File: configs/container_training.yaml

base_model: ~/models/base/CodeLlama-7b-hf
output_dir: ~/models/codellama-7b-container-v1

datasets:
  - path: ~/datasets/processed/container/container_train.json
    type: alpaca

val_set_size: 0.1

# LoRA Config
adapter: lora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Quantization
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16

# Training params
sequence_len: 1024  # Dockerfiles are typically short
micro_batch_size: 8
gradient_accumulation_steps: 4
num_epochs: 3
learning_rate: 0.0002

# Optimization
optimizer: paged_adamw_8bit
flash_attention: true
gradient_checkpointing: true
bf16: true

# MLflow
mlflow_tracking_uri: http://localhost:5000
mlflow_experiment_name: container-security-detection
```

### 5.2 Launch Container Training

```bash
axolotl train configs/container_training.yaml
```

**Expected Training Time (Single A100):**
- Dataset: 24,000 samples
- **Estimated time: 8-10 hours**

---

## 6. API Security Model Training

### 6.1 Training Configuration

```yaml
# File: configs/api_training.yaml

base_model: ~/models/base/CodeLlama-7b-Instruct-hf  # Instruct variant
output_dir: ~/models/codellama-7b-api-v1

datasets:
  - path: ~/datasets/processed/api/api_train.json
    type: alpaca

val_set_size: 0.1

# LoRA Config
adapter: lora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Quantization
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16

# Training params
sequence_len: 2048  # OpenAPI specs can be large
micro_batch_size: 4
gradient_accumulation_steps: 8
num_epochs: 3
learning_rate: 0.0002

# Optimization
optimizer: paged_adamw_8bit
flash_attention: true
gradient_checkpointing: true
bf16: true

# MLflow
mlflow_tracking_uri: http://localhost:5000
mlflow_experiment_name: api-security-detection
```

### 6.2 Launch API Training

```bash
axolotl train configs/api_training.yaml
```

**Expected Training Time (Single A100):**
- Dataset: 20,000 samples
- **Estimated time: 10-12 hours**

---

## 7. DAST Model Training

### 7.1 Training Configuration

```yaml
# File: configs/dast_training.yaml

base_model: ~/models/base/CodeLlama-7b-Instruct-hf
output_dir: ~/models/codellama-7b-dast-v1

datasets:
  - path: ~/datasets/processed/dast/dast_train.json
    type: alpaca

val_set_size: 0.1

# LoRA Config
adapter: lora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Quantization
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16

# Training params
sequence_len: 1024
micro_batch_size: 8
gradient_accumulation_steps: 4
num_epochs: 3
learning_rate: 0.0002

# Optimization
optimizer: paged_adamw_8bit
flash_attention: true
gradient_checkpointing: true
bf16: true

# MLflow
mlflow_tracking_uri: http://localhost:5000
mlflow_experiment_name: dast-payload-generation
```

### 7.2 Launch DAST Training

```bash
axolotl train configs/dast_training.yaml
```

**Expected Training Time (Single A100):**
- Dataset: 32,000 samples
- **Estimated time: 12-15 hours**

---

## 8. Model Evaluation

### 8.1 Evaluation Script

```python
# File: scripts/evaluate_model.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, base_model_path, adapter_path, test_data_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
        )
        
        # Load adapter
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.model.eval()
        
        # Load test dataset
        self.test_data = load_dataset('json', data_files=test_data_path, split='train')
    
    def generate_prediction(self, instruction, max_new_tokens=512):
        """Generate model prediction for given instruction"""
        prompt = f"<s>[INST] {instruction} [/INST]"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.95
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part
        response = response.split("[/INST]")[-1].strip()
        
        return response
    
    def evaluate_vulnerability_detection(self):
        """Evaluate vulnerability detection accuracy"""
        predictions = []
        ground_truths = []
        
        print("Evaluating vulnerability detection...")
        
        for sample in tqdm(self.test_data):
            instruction = sample['instruction']
            expected_response = sample['response']
            
            # Get prediction
            predicted_response = self.generate_prediction(instruction)
            
            # Extract vulnerability detection (simplified)
            # Check if vulnerability was detected
            has_vuln_pred = "vulnerability" in predicted_response.lower()
            has_vuln_true = "vulnerability" in expected_response.lower()
            
            predictions.append(1 if has_vuln_pred else 0)
            ground_truths.append(1 if has_vuln_true else 0)
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truths, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truths, predictions, average='binary'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print("\n=== Evaluation Metrics ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        return metrics
    
    def evaluate_cwe_classification(self):
        """Evaluate CWE classification accuracy"""
        import re
        
        correct_cwe = 0
        total_cwe = 0
        
        print("\nEvaluating CWE classification...")
        
        for sample in tqdm(self.test_data[:1000]):  # Sample 1000 for speed
            instruction = sample['instruction']
            expected_response = sample['response']
            
            # Extract expected CWE
            expected_cwe_match = re.search(r'CWE-(\d+)', expected_response)
            if not expected_cwe_match:
                continue
            
            expected_cwe = expected_cwe_match.group(0)
            
            # Get prediction
            predicted_response = self.generate_prediction(instruction)
            
            # Extract predicted CWE
            predicted_cwe_match = re.search(r'CWE-(\d+)', predicted_response)
            if predicted_cwe_match:
                predicted_cwe = predicted_cwe_match.group(0)
                
                if predicted_cwe == expected_cwe:
                    correct_cwe += 1
            
            total_cwe += 1
        
        cwe_accuracy = correct_cwe / total_cwe if total_cwe > 0 else 0
        
        print(f"\nCWE Classification Accuracy: {cwe_accuracy:.4f}")
        
        return cwe_accuracy
    
    def save_results(self, metrics, output_path):
        """Save evaluation results"""
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nResults saved to {output_path}")

# Run evaluation
if __name__ == "__main__":
    evaluator = ModelEvaluator(
        base_model_path="~/models/base/CodeLlama-7b-Python-hf",
        adapter_path="~/models/codellama-7b-sast-v1",
        test_data_path="~/datasets/processed/sast/sast_test.json"
    )
    
    metrics = evaluator.evaluate_vulnerability_detection()
    cwe_accuracy = evaluator.evaluate_cwe_classification()
    
    metrics['cwe_accuracy'] = cwe_accuracy
    
    evaluator.save_results(metrics, "~/models/codellama-7b-sast-v1/evaluation_results.json")
```

### 8.2 Run Evaluation

```bash
# Evaluate SAST model
python scripts/evaluate_model.py \
  --base_model ~/models/base/CodeLlama-7b-Python-hf \
  --adapter ~/models/codellama-7b-sast-v1 \
  --test_data ~/datasets/processed/sast/sast_test.json \
  --output ~/models/codellama-7b-sast-v1/eval_results.json

# Evaluate all models
for domain in sast sca iac container api dast; do
  echo "Evaluating $domain model..."
  python scripts/evaluate_model.py \
    --domain $domain
done
```

### 8.3 Benchmark Against Traditional Tools

```python
# File: scripts/benchmark_comparison.py

import subprocess
import json
from pathlib import Path

class BenchmarkComparison:
    def __init__(self, test_repos_path):
        self.test_repos = self._load_test_repos(test_repos_path)
    
    def run_semgrep(self, repo_path):
        """Run Semgrep and collect results"""
        result = subprocess.run(
            ['semgrep', '--config=auto', '--json', repo_path],
            capture_output=True,
            text=True
        )
        return json.loads(result.stdout)
    
    def run_ai_model(self, repo_path, model_endpoint):
        """Run AI model and collect results"""
        # Send code to AI model endpoint
        # (Implementation depends on serving setup)
        pass
    
    def compare_results(self, semgrep_results, ai_results):
        """Compare Semgrep and AI model results"""
        metrics = {
            'semgrep_findings': len(semgrep_results),
            'ai_findings': len(ai_results),
            'common_findings': 0,
            'semgrep_only': 0,
            'ai_only': 0,
            'precision_improvement': 0,
            'recall_improvement': 0
        }
        
        # Compare findings
        # (Implementation depends on result format)
        
        return metrics
    
    def run_benchmark(self):
        """Run full benchmark comparison"""
        all_metrics = []
        
        for repo in self.test_repos:
            print(f"Benchmarking {repo['name']}...")
            
            semgrep_results = self.run_semgrep(repo['path'])
            ai_results = self.run_ai_model(repo['path'], 'http://localhost:8000')
            
            metrics = self.compare_results(semgrep_results, ai_results)
            all_metrics.append(metrics)
        
        return all_metrics
```

---

## 9. Model Optimization

### 9.1 Quantization to 4-bit (GPTQ/AWQ)

```python
# File: scripts/quantize_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

def quantize_to_gptq(model_path, output_path):
    """Quantize model to 4-bit GPTQ format"""
    
    # Quantization config
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False
    )
    
    # Load model
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load calibration dataset
    from datasets import load_dataset
    calibration_data = load_dataset('json', data_files='~/datasets/processed/sast/sast_train.json', split='train[:1000]')
    
    # Prepare calibration samples
    examples = []
    for sample in calibration_data:
        text = f"<s>[INST] {sample['instruction']} [/INST] {sample['response']} </s>"
        examples.append(tokenizer(text, return_tensors='pt').input_ids)
    
    # Quantize
    model.quantize(examples)
    
    # Save quantized model
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Quantized model saved to {output_path}")

# Run quantization for all models
for domain in ['sast', 'sca', 'iac', 'container', 'api', 'dast']:
    quantize_to_gptq(
        f'~/models/codellama-7b-{domain}-v1',
        f'~/models/codellama-7b-{domain}-v1-gptq'
    )
```

### 9.2 Model Merging (Optional)

```python
# File: scripts/merge_adapter.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_adapter(base_model_path, adapter_path, output_path):
    """Merge LoRA adapter into base model"""
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge adapter weights
    merged_model = model.merge_and_unload()
    
    # Save merged model
    merged_model.save_pretrained(output_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Merged model saved to {output_path}")

# Merge all models
for domain in ['sast', 'sca', 'iac', 'container', 'api', 'dast']:
    merge_lora_adapter(
        f'~/models/base/CodeLlama-7b-hf',
        f'~/models/codellama-7b-{domain}-v1',
        f'~/models/codellama-7b-{domain}-v1-merged'
    )
```

---

## 10. Training Summary

### Training Timeline

| Model | Dataset Size | Training Time (A100) | VRAM Usage | Status |
|-------|-------------|---------------------|------------|--------|
| SAST | 80K samples | 24-30 hours | 40GB | ✅ |
| SCA | 160K samples | 20-25 hours | 35GB | ✅ |
| IaC | 40K samples | 12-15 hours | 40GB | ✅ |
| Container | 24K samples | 8-10 hours | 35GB | ✅ |
| API | 20K samples | 10-12 hours | 40GB | ✅ |
| DAST | 32K samples | 12-15 hours | 35GB | ✅ |
| **Total** | **356K samples** | **86-107 hours** | - | - |

### Cost Estimate

**AWS p4d.24xlarge (8x A100 80GB): $32.77/hour**
- Total training time (parallel): ~30 hours (training 2-3 models at once)
- **Total cost: ~$1,000**

**Budget Option - RunPod/Vast.ai (A100 80GB): $1.50-2.50/hour**
- Total training time (sequential): ~100 hours
- **Total cost: ~$150-250**

### Expected Model Performance

| Model | Target Precision | Target Recall | Target F1 | CWE Accuracy |
|-------|-----------------|---------------|-----------|--------------|
| SAST | >85% | >75% | >80% | >70% |
| SCA | >90% | >80% | >85% | N/A |
| IaC | >80% | >70% | >75% | N/A |
| Container | >85% | >75% | >80% | N/A |
| API | >80% | >70% | >75% | N/A |
| DAST | >75% | >65% | >70% | N/A |

---

## 11. Next Steps

After training is complete:

✅ **Model Evaluation** - Run comprehensive benchmarks
✅ **Quantization** - Create GPTQ/AWQ versions for faster inference
✅ **Model Deployment** - See `04_MODEL_DEPLOYMENT.md`
✅ **Integration** - See `05_INTEGRATION_ORCHESTRATION.md`

---

## Troubleshooting

### Out of Memory (OOM) Errors

```bash
# Reduce batch size
micro_batch_size: 2  # Instead of 4
gradient_accumulation_steps: 16  # Instead of 8

# Reduce sequence length
sequence_len: 1024  # Instead of 2048

# Enable more aggressive gradient checkpointing
gradient_checkpointing_kwargs:
  use_reentrant: false
```

### Slow Training

```bash
# Enable flash attention (requires compatible GPU)
flash_attention: true

# Use mixed precision
bf16: true
tf32: true

# Reduce logging frequency
logging_steps: 50  # Instead of 10
```

### Model Not Learning

```bash
# Increase learning rate
learning_rate: 0.0003  # Instead of 0.0002

# Increase LoRA rank
lora_r: 64  # Instead of 32

# Check dataset quality
python scripts/validate_dataset.py <dataset_path>
```

### Diverging Loss

```bash
# Decrease learning rate
learning_rate: 0.0001  # Instead of 0.0002

# Add gradient clipping
max_grad_norm: 0.5  # Instead of 1.0

# Increase warmup
warmup_steps: 500  # Instead of 100
```
