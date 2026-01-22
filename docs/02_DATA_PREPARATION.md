# Data Preparation Pipeline

## Overview
This guide covers the complete data preparation pipeline for training CodeLlama 7B models across all security domains (SAST, SCA, IaC, Container, DAST, API, Runtime).

---

## Table of Contents
1. [Dataset Sources & Collection](#dataset-sources--collection)
2. [Data Processing Pipeline](#data-processing-pipeline)
3. [Dataset Format Specifications](#dataset-format-specifications)
4. [Quality Validation](#quality-validation)
5. [Storage & Versioning](#storage--versioning)

---

## 1. Dataset Sources & Collection

### 1.1 SAST Dataset (Target: 100,000 samples)

#### Public Sources

**1. Juliet Test Suite (NIST SARD)**
```bash
# Download Juliet Test Suite
mkdir -p ~/datasets/sast/juliet
cd ~/datasets/sast/juliet

# C/C++ Test Cases
wget https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-c-cplusplus-v1-3.zip
unzip 2017-10-01-juliet-c-cplusplus-v1-3.zip

# Java Test Cases
wget https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-java-v1-3.zip
unzip 2017-10-01-juliet-java-v1-3.zip

# Coverage: 64,000+ test cases covering CWE Top 25
# Languages: C, C++, Java
# Format: Pairs of vulnerable and fixed code
```

**2. BigVul Dataset**
```bash
# Clone BigVul repository
cd ~/datasets/sast
git clone https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset.git bigvul

# Download full dataset
cd bigvul
wget https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset/raw/master/MSR_data_cleaned.csv

# Coverage: 10,000+ real vulnerabilities from GitHub
# Languages: C, C++
# Format: Before/after patches with CVE IDs
```

**3. Devign Dataset**
```bash
# Download Devign
mkdir -p ~/datasets/sast/devign
cd ~/datasets/sast/devign

wget https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF -O devign.json

# Coverage: 27,000 C/C++ functions
# Source: Debian, FFmpeg, QEMU, LibTIFF
# Format: Function-level with binary labels
```

**4. DiverseVul Dataset**
```bash
# Clone DiverseVul
cd ~/datasets/sast
git clone https://github.com/wagner-group/diversevul.git

cd diversevul
python scripts/download_data.py

# Coverage: 18,000+ vulnerabilities
# Languages: C, C++, Java, JavaScript, Python
# Format: Multi-language function pairs
```

**5. CVE Patches from GitHub**
```python
# Script to collect CVE fixes from GitHub
# File: scripts/collect_cve_patches.py

import requests
import os
from github import Github

# Initialize GitHub client
g = Github(os.environ['GITHUB_TOKEN'])

def collect_security_patches(language, limit=5000):
    """
    Collect security-related commits from GitHub
    """
    query = f'language:{language} CVE OR vulnerability OR security fix'
    repos = g.search_repositories(query=query, sort='stars', order='desc')
    
    patches = []
    for repo in repos[:100]:  # Top 100 repos per language
        try:
            commits = repo.get_commits()
            for commit in commits[:50]:  # Last 50 commits
                if any(keyword in commit.commit.message.lower() 
                       for keyword in ['cve', 'security', 'vulnerability', 'xss', 'sqli']):
                    patches.append({
                        'repo': repo.full_name,
                        'commit': commit.sha,
                        'message': commit.commit.message,
                        'files': commit.files,
                        'date': commit.commit.author.date
                    })
        except Exception as e:
            print(f"Error processing {repo.name}: {e}")
    
    return patches

# Collect for multiple languages
for lang in ['Python', 'JavaScript', 'Java', 'Go', 'C++']:
    patches = collect_security_patches(lang, limit=5000)
    # Save to disk
    with open(f'~/datasets/sast/github_patches_{lang.lower()}.json', 'w') as f:
        json.dump(patches, f, indent=2)
```

```bash
# Run collection script
cd ~/datasets/sast
python scripts/collect_cve_patches.py

# Expected: 20,000-30,000 patches across languages
```

**6. OWASP Benchmark**
```bash
# Clone OWASP Benchmark
cd ~/datasets/sast
git clone https://github.com/OWASP-Benchmark/BenchmarkJava.git

# Coverage: 2,740 test cases for Java
# Focus: OWASP Top 10 vulnerabilities
# Format: JUnit tests with vulnerable code
```

### 1.2 SCA Dataset (Target: 200,000 samples)

#### Vulnerability Databases

**1. National Vulnerability Database (NVD)**
```python
# Script: scripts/download_nvd.py

import requests
import json
import time
from datetime import datetime, timedelta

def download_nvd_data(start_year=2010, end_year=2024):
    """
    Download NVD CVE data
    """
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    
    all_cves = []
    
    for year in range(start_year, end_year + 1):
        print(f"Downloading CVE data for {year}...")
        
        params = {
            'pubStartDate': f'{year}-01-01T00:00:00.000',
            'pubEndDate': f'{year}-12-31T23:59:59.999'
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            cves = data.get('vulnerabilities', [])
            all_cves.extend(cves)
            print(f"  Downloaded {len(cves)} CVEs")
        else:
            print(f"  Error: {response.status_code}")
        
        time.sleep(6)  # Rate limit: 10 requests per minute
    
    # Save to file
    with open('~/datasets/sca/nvd_cves.json', 'w') as f:
        json.dump(all_cves, f, indent=2)
    
    return all_cves

# Run download
cves = download_nvd_data()
print(f"Total CVEs: {len(cves)}")
```

**2. OSV Database**
```bash
# Clone OSV dataset
cd ~/datasets/sca
git clone https://github.com/google/osv.dev.git

# Download ecosystems
cd osv.dev
mkdir -p data

# Download PyPI vulnerabilities
gsutil -m cp -r gs://osv-vulnerabilities/PyPI ./data/

# Download npm vulnerabilities
gsutil -m cp -r gs://osv-vulnerabilities/npm ./data/

# Download Maven vulnerabilities
gsutil -m cp -r gs://osv-vulnerabilities/Maven ./data/

# Download Go vulnerabilities
gsutil -m cp -r gs://osv-vulnerabilities/Go ./data/

# Coverage: 500,000+ vulnerabilities across ecosystems
```

**3. GitHub Security Advisories**
```python
# Script: scripts/download_github_advisories.py

from github import Github
import os
import json

g = Github(os.environ['GITHUB_TOKEN'])

def download_advisories(ecosystem='npm', limit=10000):
    """
    Download GitHub Security Advisories
    """
    advisories = []
    
    # GraphQL query
    query = """
    query($cursor: String) {
      securityAdvisories(first: 100, after: $cursor, ecosystem: %s) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          ghsaId
          summary
          description
          severity
          publishedAt
          updatedAt
          vulnerabilities(first: 10) {
            nodes {
              package { name ecosystem }
              vulnerableVersionRange
              firstPatchedVersion { identifier }
            }
          }
        }
      }
    }
    """ % ecosystem
    
    # Execute query and collect all advisories
    # (Implementation details omitted for brevity)
    
    return advisories

# Download for all ecosystems
for ecosystem in ['npm', 'pip', 'maven', 'nuget', 'rubygems', 'go']:
    advisories = download_advisories(ecosystem)
    with open(f'~/datasets/sca/github_advisories_{ecosystem}.json', 'w') as f:
        json.dump(advisories, f, indent=2)
```

**4. Dependency Manifests with Known Vulnerabilities**
```python
# Script: scripts/collect_vulnerable_manifests.py

import os
import json
import requests
from github import Github

g = Github(os.environ['GITHUB_TOKEN'])

def collect_package_files(language, package_file, limit=10000):
    """
    Collect package manifest files from GitHub
    """
    query = f'filename:{package_file} language:{language}'
    
    code_files = g.search_code(query=query)
    
    manifests = []
    for file in code_files[:limit]:
        try:
            content = file.decoded_content.decode('utf-8')
            manifests.append({
                'repo': file.repository.full_name,
                'path': file.path,
                'content': content,
                'url': file.html_url
            })
        except Exception as e:
            print(f"Error: {e}")
    
    return manifests

# Collect package files
configs = [
    ('JavaScript', 'package.json'),
    ('Python', 'requirements.txt'),
    ('Python', 'Pipfile'),
    ('Java', 'pom.xml'),
    ('Java', 'build.gradle'),
    ('Ruby', 'Gemfile'),
    ('Go', 'go.mod'),
    ('C#', 'packages.config'),
]

for lang, filename in configs:
    print(f"Collecting {filename} files...")
    manifests = collect_package_files(lang, filename, limit=5000)
    
    with open(f'~/datasets/sca/manifests_{filename.replace(".", "_")}.json', 'w') as f:
        json.dump(manifests, f, indent=2)
```

### 1.3 IaC Security Dataset (Target: 50,000 samples)

**1. Terraform Misconfigurations**
```bash
# Collect Terraform files from GitHub
cd ~/datasets/iac

# Search for Terraform files
gh search code --language=HCL "resource aws_s3_bucket" --limit 1000 > s3_configs.txt
gh search code --language=HCL "resource aws_security_group" --limit 1000 > sg_configs.txt
gh search code --language=HCL "resource aws_iam" --limit 1000 > iam_configs.txt

# Or use GitHub API
python scripts/collect_terraform_files.py
```

**2. Checkov/tfsec Rule Violations**
```python
# Script: scripts/generate_iac_dataset.py

import os
import subprocess
import json

def scan_terraform_repos(repo_list_file):
    """
    Scan Terraform repositories and collect findings
    """
    with open(repo_list_file) as f:
        repos = [line.strip() for line in f]
    
    dataset = []
    
    for repo in repos:
        # Clone repo
        repo_name = repo.split('/')[-1]
        subprocess.run(['git', 'clone', f'https://github.com/{repo}', f'/tmp/{repo_name}'])
        
        # Run checkov
        result = subprocess.run(
            ['checkov', '-d', f'/tmp/{repo_name}', '-o', 'json'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            findings = json.loads(result.stdout)
            
            for check in findings.get('results', {}).get('failed_checks', []):
                dataset.append({
                    'repo': repo,
                    'file': check['file_path'],
                    'code': check.get('code_block', ''),
                    'check_id': check['check_id'],
                    'check_name': check['check_name'],
                    'severity': check.get('severity', 'UNKNOWN'),
                    'guideline': check.get('guideline', '')
                })
        
        # Cleanup
        subprocess.run(['rm', '-rf', f'/tmp/{repo_name}'])
    
    return dataset

# Process repos
dataset = scan_terraform_repos('terraform_repos.txt')

with open('~/datasets/iac/checkov_findings.json', 'w') as f:
    json.dump(dataset, f, indent=2)
```

**3. CIS Benchmarks**
```bash
# Download CIS Benchmarks test cases
cd ~/datasets/iac

# AWS CIS Benchmark examples
wget https://raw.githubusercontent.com/aquasecurity/defsec/master/test/rego/aws_test.rego

# Use kube-bench for Kubernetes
git clone https://github.com/aquasecurity/kube-bench.git
```

### 1.4 Container Security Dataset (Target: 30,000 samples)

**1. Vulnerable Dockerfiles**
```python
# Script: scripts/collect_dockerfiles.py

from github import Github
import os

g = Github(os.environ['GITHUB_TOKEN'])

def collect_dockerfiles(limit=20000):
    """
    Collect Dockerfiles from GitHub
    """
    query = 'filename:Dockerfile'
    
    dockerfiles = []
    code_files = g.search_code(query=query)
    
    for file in code_files[:limit]:
        try:
            content = file.decoded_content.decode('utf-8')
            dockerfiles.append({
                'repo': file.repository.full_name,
                'path': file.path,
                'content': content,
                'stars': file.repository.stargazers_count
            })
        except Exception as e:
            print(f"Error: {e}")
    
    return dockerfiles

dockerfiles = collect_dockerfiles()

with open('~/datasets/container/dockerfiles.json', 'w') as f:
    json.dump(dockerfiles, f, indent=2)
```

**2. Scan with Trivy**
```bash
# Scan popular container images and collect findings
cd ~/datasets/container

cat <<EOF > scan_images.sh
#!/bin/bash

IMAGES=(
  "python:3.9"
  "python:3.8"
  "node:14"
  "node:16"
  "ubuntu:18.04"
  "ubuntu:20.04"
  "nginx:1.19"
  "nginx:1.21"
  "postgres:12"
  "postgres:13"
  "redis:6"
  "mysql:5.7"
)

for image in "\${IMAGES[@]}"; do
  echo "Scanning \$image..."
  trivy image --format json --output "trivy_\${image//[:\/]/_}.json" "\$image"
done
EOF

chmod +x scan_images.sh
./scan_images.sh
```

### 1.5 API Security Dataset (Target: 25,000 samples)

**1. OpenAPI Specifications**
```bash
# Clone APIs-guru repository (largest collection of OpenAPI specs)
cd ~/datasets/api
git clone https://github.com/APIs-guru/openapi-directory.git

# Coverage: 2,000+ real-world API specifications
```

**2. API Security Test Cases**
```python
# Download OWASP API Security Top 10 test cases
cd ~/datasets/api
git clone https://github.com/OWASP/API-Security.git

# PortSwigger Web Security Academy API labs
# (Manual collection from https://portswigger.net/web-security/api-testing)
```

### 1.6 DAST/Penetration Testing Dataset (Target: 40,000 samples)

**1. Exploit Database**
```bash
# Clone Exploit-DB
cd ~/datasets/dast
git clone https://github.com/offensive-security/exploitdb.git

# Coverage: 50,000+ exploits
```

**2. Metasploit Modules**
```bash
# Clone Metasploit Framework
cd ~/datasets/dast
git clone https://github.com/rapid7/metasploit-framework.git

# Coverage: 2,000+ exploit modules
```

**3. DVWA, WebGoat, Juice Shop**
```bash
# Clone vulnerable applications for training data
cd ~/datasets/dast

git clone https://github.com/digininja/DVWA.git
git clone https://github.com/WebGoat/WebGoat.git
git clone https://github.com/juice-shop/juice-shop.git

# Run applications and collect attack patterns
```

---

## 2. Data Processing Pipeline

### 2.1 ETL Pipeline Architecture

```python
# File: scripts/etl_pipeline.py

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import hashlib

class DataProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract(self, source: str) -> List[Dict]:
        """Extract data from various sources"""
        raise NotImplementedError
    
    def transform(self, data: List[Dict]) -> List[Dict]:
        """Clean, normalize, and augment data"""
        raise NotImplementedError
    
    def load(self, data: List[Dict], filename: str):
        """Save processed data"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(data)} samples to {output_path}")
    
    def deduplicate(self, data: List[Dict], key_field: str = 'code') -> List[Dict]:
        """Remove duplicate entries"""
        seen = set()
        unique_data = []
        
        for item in data:
            # Create hash of key field
            content_hash = hashlib.md5(item[key_field].encode()).hexdigest()
            
            if content_hash not in seen:
                seen.add(content_hash)
                unique_data.append(item)
        
        print(f"Removed {len(data) - len(unique_data)} duplicates")
        return unique_data
```

### 2.2 SAST Data Processor

```python
# File: scripts/process_sast_data.py

import re
import json
from transformers import AutoTokenizer

class SASTDataProcessor(DataProcessor):
    def __init__(self, input_dir, output_dir):
        super().__init__(input_dir, output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
        self.max_length = 2048
    
    def process_juliet(self, juliet_dir: str):
        """Process Juliet Test Suite"""
        samples = []
        
        # Find all test case files
        for cwe_dir in Path(juliet_dir).glob('CWE*'):
            cwe_id = cwe_dir.name
            
            for test_file in cwe_dir.rglob('*.c'):
                with open(test_file) as f:
                    code = f.read()
                
                # Determine if vulnerable or patched
                is_vulnerable = 'bad' in test_file.stem.lower()
                
                # Extract vulnerability type
                vuln_match = re.search(r'CWE(\d+)', cwe_id)
                cwe_number = vuln_match.group(1) if vuln_match else 'Unknown'
                
                samples.append({
                    'code': code,
                    'language': 'c',
                    'vulnerable': is_vulnerable,
                    'cwe_id': f'CWE-{cwe_number}',
                    'source': 'juliet',
                    'file_path': str(test_file)
                })
        
        return samples
    
    def process_bigvul(self, csv_path: str):
        """Process BigVul dataset"""
        df = pd.read_csv(csv_path)
        samples = []
        
        for _, row in df.iterrows():
            if pd.isna(row['code']):
                continue
            
            samples.append({
                'code': row['code'],
                'language': 'c',
                'vulnerable': True,
                'cve_id': row.get('cve_id', ''),
                'cwe_id': row.get('cwe_id', ''),
                'commit_message': row.get('commit_message', ''),
                'source': 'bigvul'
            })
        
        return samples
    
    def create_training_format(self, samples: List[Dict]) -> List[Dict]:
        """Convert to instruction-tuning format"""
        formatted = []
        
        for sample in samples:
            # Skip if code is too long
            tokens = self.tokenizer.encode(sample['code'])
            if len(tokens) > self.max_length - 500:  # Reserve space for prompt
                continue
            
            if sample['vulnerable']:
                instruction = f"""You are a security code analyzer. Analyze the following code for vulnerabilities.

Language: {sample['language']}
Code:
```{sample['language']}
{sample['code']}
```

Identify any security vulnerabilities, provide the CWE type, severity, and explanation."""
                
                response = f"""Vulnerability found:
- Type: {self._get_vulnerability_name(sample.get('cwe_id', 'Unknown'))}
- CWE: {sample.get('cwe_id', 'Unknown')}
- Severity: {self._estimate_severity(sample.get('cwe_id', ''))}
- Description: {self._generate_description(sample)}
- Recommendation: {self._generate_recommendation(sample)}"""
            
            else:
                instruction = f"""You are a security code analyzer. Analyze the following code for vulnerabilities.

Language: {sample['language']}
Code:
```{sample['language']}
{sample['code']}
```

Identify any security vulnerabilities, provide the CWE type, severity, and explanation."""
                
                response = "No vulnerabilities detected. The code appears secure."
            
            formatted.append({
                'instruction': instruction,
                'response': response,
                'metadata': {
                    'source': sample['source'],
                    'cwe_id': sample.get('cwe_id', ''),
                    'language': sample['language']
                }
            })
        
        return formatted
    
    def _get_vulnerability_name(self, cwe_id: str) -> str:
        """Map CWE ID to vulnerability name"""
        cwe_map = {
            'CWE-89': 'SQL Injection',
            'CWE-79': 'Cross-Site Scripting (XSS)',
            'CWE-78': 'OS Command Injection',
            'CWE-119': 'Buffer Overflow',
            'CWE-120': 'Buffer Copy without Checking Size',
            'CWE-416': 'Use After Free',
            'CWE-787': 'Out-of-bounds Write',
            # Add more mappings
        }
        return cwe_map.get(cwe_id, 'Unknown Vulnerability')
    
    def _estimate_severity(self, cwe_id: str) -> str:
        """Estimate severity based on CWE"""
        critical = ['CWE-89', 'CWE-78', 'CWE-79']
        high = ['CWE-119', 'CWE-120', 'CWE-416']
        
        if cwe_id in critical:
            return 'CRITICAL'
        elif cwe_id in high:
            return 'HIGH'
        else:
            return 'MEDIUM'
    
    def _generate_description(self, sample: Dict) -> str:
        """Generate vulnerability description"""
        cwe_id = sample.get('cwe_id', '')
        
        descriptions = {
            'CWE-89': 'User input is not properly sanitized before being used in SQL query, allowing SQL injection attacks.',
            'CWE-79': 'User input is reflected in HTML output without proper encoding, allowing cross-site scripting attacks.',
            # Add more descriptions
        }
        
        return descriptions.get(cwe_id, 'Security vulnerability detected in code.')
    
    def _generate_recommendation(self, sample: Dict) -> str:
        """Generate remediation advice"""
        cwe_id = sample.get('cwe_id', '')
        
        recommendations = {
            'CWE-89': 'Use parameterized queries or prepared statements instead of string concatenation.',
            'CWE-79': 'Apply proper output encoding (HTML entity encoding) before rendering user input.',
            # Add more recommendations
        }
        
        return recommendations.get(cwe_id, 'Follow secure coding best practices.')

# Run processor
processor = SASTDataProcessor('~/datasets/sast', '~/datasets/processed/sast')

# Process all sources
juliet_samples = processor.process_juliet('~/datasets/sast/juliet')
bigvul_samples = processor.process_bigvul('~/datasets/sast/bigvul/MSR_data_cleaned.csv')

# Combine all samples
all_samples = juliet_samples + bigvul_samples

# Deduplicate
unique_samples = processor.deduplicate(all_samples, key_field='code')

# Create training format
training_data = processor.create_training_format(unique_samples)

# Split into train/val/test
from sklearn.model_selection import train_test_split

train, temp = train_test_split(training_data, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save datasets
processor.load(train, 'sast_train.json')
processor.load(val, 'sast_val.json')
processor.load(test, 'sast_test.json')

print(f"Training samples: {len(train)}")
print(f"Validation samples: {len(val)}")
print(f"Test samples: {len(test)}")
```

### 2.3 SCA Data Processor

```python
# File: scripts/process_sca_data.py

class SCADataProcessor(DataProcessor):
    def process_nvd(self, nvd_file: str):
        """Process NVD CVE data"""
        with open(nvd_file) as f:
            cves = json.load(f)
        
        samples = []
        
        for vuln in cves:
            cve = vuln.get('cve', {})
            cve_id = cve.get('id', '')
            
            # Extract package info from CPE
            configurations = cve.get('configurations', [])
            affected_packages = []
            
            for config in configurations:
                for node in config.get('nodes', []):
                    for cpe_match in node.get('cpeMatch', []):
                        if cpe_match.get('vulnerable'):
                            affected_packages.append({
                                'cpe': cpe_match.get('criteria', ''),
                                'version_start': cpe_match.get('versionStartIncluding'),
                                'version_end': cpe_match.get('versionEndExcluding')
                            })
            
            # Get CVSS score
            metrics = cve.get('metrics', {})
            cvss_v3 = metrics.get('cvssMetricV31', [{}])[0].get('cvssData', {})
            
            samples.append({
                'cve_id': cve_id,
                'description': cve.get('descriptions', [{}])[0].get('value', ''),
                'severity': cvss_v3.get('baseSeverity', 'UNKNOWN'),
                'cvss_score': cvss_v3.get('baseScore', 0.0),
                'affected_packages': affected_packages,
                'references': [ref.get('url') for ref in cve.get('references', [])],
                'published_date': cve.get('published', ''),
                'cwe_ids': [w.get('description', [{}])[0].get('value', '') 
                           for w in cve.get('weaknesses', [])]
            })
        
        return samples
    
    def create_manifest_vulnerability_pairs(self, manifests: List[Dict], cves: List[Dict]):
        """Create training pairs of manifests and their vulnerabilities"""
        training_data = []
        
        for manifest in manifests:
            content = manifest['content']
            
            # Parse manifest to extract package names/versions
            packages = self._parse_manifest(content, manifest.get('type', 'package.json'))
            
            # Match with CVEs
            vulnerable_packages = []
            for pkg in packages:
                matching_cves = self._find_cves_for_package(pkg, cves)
                if matching_cves:
                    vulnerable_packages.append({
                        'package': pkg,
                        'cves': matching_cves
                    })
            
            # Create instruction format
            if vulnerable_packages:
                instruction = f"""Analyze the following dependency manifest for vulnerable packages.

Manifest:
```
{content}
```

Identify vulnerable dependencies, provide CVE IDs, severity scores, and remediation advice."""
                
                response = "Vulnerable dependencies found:\n\n"
                for vuln_pkg in vulnerable_packages:
                    pkg = vuln_pkg['package']
                    response += f"- {pkg['name']} ({pkg['version']})\n"
                    for cve in vuln_pkg['cves'][:3]:  # Limit to top 3 CVEs
                        response += f"  - {cve['cve_id']} ({cve['severity']}) - {cve['description'][:100]}...\n"
                        response += f"    Fix: Upgrade to version {cve.get('fixed_version', 'latest')}\n"
                
                training_data.append({
                    'instruction': instruction,
                    'response': response,
                    'metadata': {
                        'manifest_type': manifest.get('type'),
                        'vulnerable_count': len(vulnerable_packages)
                    }
                })
        
        return training_data
    
    def _parse_manifest(self, content: str, manifest_type: str) -> List[Dict]:
        """Parse manifest file to extract packages"""
        packages = []
        
        if manifest_type == 'package.json':
            try:
                data = json.loads(content)
                deps = {**data.get('dependencies', {}), **data.get('devDependencies', {})}
                for name, version in deps.items():
                    packages.append({
                        'name': name,
                        'version': version.lstrip('^~'),
                        'ecosystem': 'npm'
                    })
            except:
                pass
        
        elif manifest_type == 'requirements.txt':
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    match = re.match(r'([a-zA-Z0-9_-]+)\s*([=<>!]+)\s*([0-9.]+)', line)
                    if match:
                        packages.append({
                            'name': match.group(1),
                            'version': match.group(3),
                            'ecosystem': 'pypi'
                        })
        
        # Add more manifest types (pom.xml, build.gradle, etc.)
        
        return packages
    
    def _find_cves_for_package(self, package: Dict, cves: List[Dict]) -> List[Dict]:
        """Find CVEs affecting a specific package"""
        # Simplified matching logic
        # In production, use proper version range matching
        matching_cves = []
        
        for cve in cves:
            # Check if CVE affects this package
            # (Implementation depends on CVE data structure)
            pass
        
        return matching_cves
```

### 2.4 Data Augmentation

```python
# File: scripts/data_augmentation.py

class CodeAugmentor:
    """Augment code samples to increase dataset diversity"""
    
    def augment_code(self, code: str, language: str) -> List[str]:
        """Generate variations of code snippet"""
        variations = [code]  # Original
        
        # Variable renaming
        variations.append(self._rename_variables(code, language))
        
        # Comment injection/removal
        variations.append(self._modify_comments(code))
        
        # Whitespace variations
        variations.append(self._modify_whitespace(code))
        
        # Code style variations (brackets, indentation)
        variations.append(self._modify_style(code, language))
        
        return variations
    
    def _rename_variables(self, code: str, language: str) -> str:
        """Rename variables consistently"""
        # Simple implementation - use AST for better results
        import re
        
        # Find common variable names
        var_pattern = r'\b(temp|tmp|result|data|value|user|input)\b'
        
        replacements = {
            'temp': 'temporary',
            'tmp': 'tempVar',
            'result': 'output',
            'data': 'information',
            'value': 'val',
            'user': 'usr',
            'input': 'userInput'
        }
        
        modified = code
        for old, new in replacements.items():
            modified = re.sub(rf'\b{old}\b', new, modified)
        
        return modified
    
    def _modify_comments(self, code: str) -> str:
        """Add or remove comments"""
        # Remove existing comments
        import re
        no_comments = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        no_comments = re.sub(r'/\*.*?\*/', '', no_comments, flags=re.DOTALL)
        return no_comments
    
    def _modify_whitespace(self, code: str) -> str:
        """Modify whitespace while preserving semantics"""
        # Compress whitespace
        import re
        compressed = re.sub(r'\n\s*\n', '\n', code)
        compressed = re.sub(r'  +', ' ', compressed)
        return compressed
    
    def _modify_style(self, code: str, language: str) -> str:
        """Modify code style"""
        # For C/Java: convert K&R to Allman style or vice versa
        # Simplified implementation
        return code
```

---

## 3. Dataset Format Specifications

### 3.1 Training Data Format (JSONL)

```json
{
  "instruction": "You are a security code analyzer. Analyze the following code for vulnerabilities.\n\nLanguage: python\nCode:\n```python\ndef login(username, password):\n    query = \"SELECT * FROM users WHERE username='\" + username + \"' AND password='\" + password + \"'\"\n    cursor.execute(query)\n```\n\nIdentify any security vulnerabilities, provide the CWE type, severity, and explanation.",
  "response": "Vulnerability found:\n- Type: SQL Injection\n- CWE: CWE-89\n- Severity: CRITICAL\n- Description: User input (username and password) is directly concatenated into SQL query without sanitization, allowing SQL injection attacks.\n- Recommendation: Use parameterized queries or prepared statements instead of string concatenation.",
  "metadata": {
    "source": "synthetic",
    "cwe_id": "CWE-89",
    "language": "python",
    "difficulty": "easy"
  }
}
```

### 3.2 Dataset Statistics Tracking

```python
# File: scripts/dataset_stats.py

def calculate_dataset_statistics(dataset_path: str):
    """Calculate and display dataset statistics"""
    with open(dataset_path) as f:
        data = json.load(f)
    
    stats = {
        'total_samples': len(data),
        'languages': {},
        'cwe_types': {},
        'severities': {},
        'sources': {},
        'avg_code_length': 0,
        'avg_tokens': 0
    }
    
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    total_length = 0
    total_tokens = 0
    
    for sample in data:
        metadata = sample.get('metadata', {})
        
        # Count languages
        lang = metadata.get('language', 'unknown')
        stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
        
        # Count CWE types
        cwe = metadata.get('cwe_id', 'unknown')
        stats['cwe_types'][cwe] = stats['cwe_types'].get(cwe, 0) + 1
        
        # Count severities
        severity = metadata.get('severity', 'unknown')
        stats['severities'][severity] = stats['severities'].get(severity, 0) + 1
        
        # Count sources
        source = metadata.get('source', 'unknown')
        stats['sources'][source] = stats['sources'].get(source, 0) + 1
        
        # Calculate lengths
        instruction = sample.get('instruction', '')
        total_length += len(instruction)
        total_tokens += len(tokenizer.encode(instruction))
    
    stats['avg_code_length'] = total_length / len(data)
    stats['avg_tokens'] = total_tokens / len(data)
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"\nLanguages:")
    for lang, count in sorted(stats['languages'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {count} ({count/stats['total_samples']*100:.1f}%)")
    
    print(f"\nCWE Types:")
    for cwe, count in sorted(stats['cwe_types'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {cwe}: {count} ({count/stats['total_samples']*100:.1f}%)")
    
    print(f"\nSources:")
    for source, count in stats['sources'].items():
        print(f"  {source}: {count} ({count/stats['total_samples']*100:.1f}%)")
    
    print(f"\nAverage code length: {stats['avg_code_length']:.0f} characters")
    print(f"Average tokens: {stats['avg_tokens']:.0f} tokens")
    
    return stats
```

---

## 4. Quality Validation

### 4.1 Automated Validation

```python
# File: scripts/validate_dataset.py

class DatasetValidator:
    def __init__(self, max_length=2048):
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
        self.max_length = max_length
        self.errors = []
    
    def validate(self, dataset_path: str) -> bool:
        """Validate dataset quality"""
        with open(dataset_path) as f:
            data = json.load(f)
        
        print(f"Validating {len(data)} samples...")
        
        valid_samples = 0
        
        for idx, sample in enumerate(data):
            if self._validate_sample(sample, idx):
                valid_samples += 1
        
        print(f"\nValidation Results:")
        print(f"  Valid samples: {valid_samples}/{len(data)} ({valid_samples/len(data)*100:.1f}%)")
        print(f"  Errors found: {len(self.errors)}")
        
        if self.errors:
            print("\nTop 10 errors:")
            for error in self.errors[:10]:
                print(f"  {error}")
        
        return valid_samples == len(data)
    
    def _validate_sample(self, sample: Dict, idx: int) -> bool:
        """Validate individual sample"""
        is_valid = True
        
        # Check required fields
        if 'instruction' not in sample:
            self.errors.append(f"Sample {idx}: Missing 'instruction' field")
            is_valid = False
        
        if 'response' not in sample:
            self.errors.append(f"Sample {idx}: Missing 'response' field")
            is_valid = False
        
        # Check token length
        if 'instruction' in sample:
            tokens = self.tokenizer.encode(sample['instruction'])
            if len(tokens) > self.max_length:
                self.errors.append(f"Sample {idx}: Instruction too long ({len(tokens)} tokens)")
                is_valid = False
        
        # Check for empty content
        if sample.get('instruction', '').strip() == '':
            self.errors.append(f"Sample {idx}: Empty instruction")
            is_valid = False
        
        if sample.get('response', '').strip() == '':
            self.errors.append(f"Sample {idx}: Empty response")
            is_valid = False
        
        # Check metadata
        if 'metadata' not in sample:
            self.errors.append(f"Sample {idx}: Missing metadata")
            is_valid = False
        
        return is_valid

# Run validation
validator = DatasetValidator()
validator.validate('~/datasets/processed/sast/sast_train.json')
```

### 4.2 Manual Review Process

```markdown
## Manual Review Checklist

For a random sample of 1000 entries:

1. **Code Quality**
   - [ ] Code is syntactically valid
   - [ ] Code is representative of real-world code
   - [ ] Vulnerability is actually present (for vulnerable samples)

2. **Label Accuracy**
   - [ ] CWE classification is correct
   - [ ] Severity rating is appropriate
   - [ ] Description accurately describes the vulnerability

3. **Response Quality**
   - [ ] Remediation advice is correct
   - [ ] Recommendation is actionable
   - [ ] No hallucinated information

4. **Diversity**
   - [ ] Multiple languages represented
   - [ ] Various vulnerability types
   - [ ] Different code complexities

5. **Bias Check**
   - [ ] No overrepresentation of specific CWEs
   - [ ] Balanced vulnerable/safe ratio
   - [ ] Diverse code patterns
```

---

## 5. Storage & Versioning

### 5.1 Upload to Object Storage

```python
# File: scripts/upload_datasets.py

from minio import Minio
from minio.error import S3Error
import os

def upload_to_minio(local_dir: str, bucket: str = 'datasets'):
    """Upload processed datasets to MinIO"""
    
    # Initialize MinIO client
    client = Minio(
        "minio:9000",  # MinIO service in Kubernetes
        access_key=os.environ['MINIO_ROOT_USER'],
        secret_key=os.environ['MINIO_ROOT_PASSWORD'],
        secure=False
    )
    
    # Create bucket if not exists
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    
    # Upload all files in directory
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            object_name = os.path.relpath(local_path, local_dir)
            
            print(f"Uploading {object_name}...")
            client.fput_object(bucket, object_name, local_path)
    
    print(f"Upload complete!")

# Upload all processed datasets
upload_to_minio('~/datasets/processed', 'datasets')
```

### 5.2 Dataset Versioning

```bash
# Create dataset versions with DVC (Data Version Control)

# Install DVC
pip install dvc dvc-s3

# Initialize DVC
cd ~/datasets
dvc init

# Configure MinIO as remote storage
dvc remote add -d minio s3://datasets
dvc remote modify minio endpointurl http://minio:9000
dvc remote modify minio access_key_id $MINIO_ROOT_USER
dvc remote modify minio secret_access_key $MINIO_ROOT_PASSWORD

# Track datasets
dvc add processed/sast/sast_train.json
dvc add processed/sast/sast_val.json
dvc add processed/sast/sast_test.json

# Commit to Git
git add processed/sast/*.dvc .gitignore
git commit -m "Add SAST training datasets v1.0"

# Push to remote storage
dvc push

# Tag version
git tag -a datasets-v1.0 -m "Initial SAST datasets"
git push origin datasets-v1.0
```

### 5.3 Dataset Catalog

```yaml
# File: dataset_catalog.yaml

datasets:
  sast_v1:
    name: "SAST Training Dataset v1.0"
    description: "Static Application Security Testing training data"
    created_date: "2024-01-15"
    samples:
      train: 80000
      val: 10000
      test: 10000
    languages:
      - python
      - javascript
      - java
      - c
      - cpp
      - go
    cwe_coverage:
      - CWE-89
      - CWE-79
      - CWE-78
      - CWE-119
      # ... more CWEs
    sources:
      - juliet
      - bigvul
      - devign
      - github_cves
    location: "s3://datasets/sast/v1/"
    format: "jsonl"
    
  sca_v1:
    name: "SCA Training Dataset v1.0"
    description: "Software Composition Analysis training data"
    created_date: "2024-01-20"
    samples:
      train: 160000
      val: 20000
      test: 20000
    ecosystems:
      - npm
      - pypi
      - maven
      - nuget
      - rubygems
    sources:
      - nvd
      - osv
      - github_advisories
    location: "s3://datasets/sca/v1/"
    format: "jsonl"
```

---

## 6. Execution Plan

### Week-by-Week Plan

**Week 1: Setup & Collection**
```bash
# Day 1-2: Setup environment
mkdir -p ~/datasets/{sast,sca,iac,container,api,dast}/raw
pip install -r scripts/requirements.txt

# Day 3-5: Download public datasets
bash scripts/download_all_datasets.sh

# Day 6-7: Collect GitHub data
python scripts/collect_cve_patches.py
python scripts/collect_dockerfiles.py
python scripts/collect_terraform_files.py
```

**Week 2: Processing SAST & SCA**
```bash
# Process SAST data
python scripts/process_sast_data.py

# Process SCA data  
python scripts/process_sca_data.py

# Validate datasets
python scripts/validate_dataset.py ~/datasets/processed/sast/sast_train.json
python scripts/validate_dataset.py ~/datasets/processed/sca/sca_train.json
```

**Week 3: Processing IaC, Container, API**
```bash
# Process IaC data
python scripts/process_iac_data.py

# Process Container data
python scripts/process_container_data.py

# Process API data
python scripts/process_api_data.py
```

**Week 4: Finalization**
```bash
# Calculate statistics
python scripts/dataset_stats.py

# Upload to MinIO
python scripts/upload_datasets.py

# Version with DVC
dvc add processed/**/*.json
git add processed/**/*.dvc
git commit -m "Add all training datasets v1.0"
dvc push
```

---

## Summary

✅ **SAST Dataset**: 100K samples (CWE-based vulnerabilities)
✅ **SCA Dataset**: 200K samples (CVE + package vulnerabilities)
✅ **IaC Dataset**: 50K samples (misconfigurations)
✅ **Container Dataset**: 30K samples (Dockerfile + image vulnerabilities)
✅ **API Dataset**: 25K samples (OpenAPI specs + vulnerabilities)
✅ **DAST Dataset**: 40K samples (exploits + attack patterns)

**Total**: ~445,000 training samples

**Storage Required**: ~50GB (compressed)

**Processing Time**: 3-4 weeks

**Next Step**: Model Training (see `03_MODEL_TRAINING.md`)
