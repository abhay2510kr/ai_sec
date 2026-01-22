#!/usr/bin/env python3
"""
Convert CVE JSON files (2024-2025) to SCA training dataset
Processes ~40,000 CVEs into instruction-response format for CodeLlama fine-tuning
"""

import json
import glob
from pathlib import Path
from typing import List, Dict
import random

def load_cve_file(filepath: str) -> Dict:
    """Load a single CVE JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def extract_package_info(cve_data: Dict) -> List[Dict]:
    """Extract package information from CVE"""
    packages = []
    
    try:
        containers = cve_data.get('containers', {})
        cna = containers.get('cna', {})
        affected = cna.get('affected', [])
        
        for item in affected:
            vendor = item.get('vendor', 'unknown')
            product = item.get('product', 'unknown')
            versions = item.get('versions', [])
            
            # Skip if no useful package info
            if vendor == 'n/a' or product == 'n/a':
                continue
                
            for ver in versions:
                version = ver.get('version', 'unknown')
                status = ver.get('status', 'affected')
                
                if status == 'affected':
                    packages.append({
                        'vendor': vendor,
                        'product': product,
                        'version': version
                    })
    except:
        pass
    
    return packages

def get_severity(cve_data: Dict) -> str:
    """Extract CVSS severity"""
    try:
        containers = cve_data.get('containers', {})
        cna = containers.get('cna', {})
        metrics = cna.get('metrics', [])
        
        for metric in metrics:
            if 'cvssV3_1' in metric:
                return metric['cvssV3_1'].get('baseSeverity', 'MEDIUM')
            elif 'cvssV3_0' in metric:
                return metric['cvssV3_0'].get('baseSeverity', 'MEDIUM')
    except:
        pass
    
    return 'MEDIUM'

def get_description(cve_data: Dict) -> str:
    """Extract vulnerability description"""
    try:
        containers = cve_data.get('containers', {})
        cna = containers.get('cna', {})
        descriptions = cna.get('descriptions', [])
        
        for desc in descriptions:
            if desc.get('lang') == 'en':
                return desc.get('value', '')[:300]  # Limit length
    except:
        pass
    
    return "Vulnerability information not available"

def create_package_manifest(packages: List[Dict], ecosystem: str = 'npm') -> str:
    """Create a mock package manifest"""
    if not packages:
        return ""
    
    if ecosystem == 'npm':
        manifest = {
            "name": "vulnerable-app",
            "dependencies": {}
        }
        for pkg in packages[:5]:  # Max 5 packages
            pkg_name = pkg['product'].replace('_', '-')
            version = pkg.get('version', '1.0.0')
            if version not in ['n/a', 'unknown', '*']:
                manifest['dependencies'][pkg_name] = version
        
        return json.dumps(manifest, indent=2)
    
    elif ecosystem == 'pip':
        lines = []
        for pkg in packages[:5]:
            pkg_name = pkg['product']
            version = pkg.get('version', '1.0.0')
            if version not in ['n/a', 'unknown', '*']:
                lines.append(f"{pkg_name}=={version}")
        return '\n'.join(lines)
    
    return ""

def create_training_example(cve_file: str) -> Dict:
    """Convert CVE file to training example"""
    cve_data = load_cve_file(cve_file)
    if not cve_data:
        return None
    
    # Extract CVE ID
    cve_metadata = cve_data.get('cveMetadata', {})
    cve_id = cve_metadata.get('cveId', 'Unknown')
    
    # Extract package info
    packages = extract_package_info(cve_data)
    if not packages:
        return None  # Skip if no package info
    
    # Get severity and description
    severity = get_severity(cve_data)
    description = get_description(cve_data)
    
    # Random ecosystem
    ecosystem = random.choice(['npm', 'pip'])
    manifest = create_package_manifest(packages, ecosystem)
    
    if not manifest:
        return None
    
    # Create instruction
    instruction = f"Analyze this {ecosystem} dependency manifest for known vulnerabilities"
    
    # Create input
    format_type = "json" if ecosystem == "npm" else "txt"
    input_text = f"```{format_type}\n{manifest}\n```"
    
    # Create response
    response = f"""Vulnerabilities found:

1. **{cve_id}** ({severity})
   - Affected packages: {', '.join(pkg['product'] for pkg in packages[:3])}
   - Description: {description}
   - Recommendation: Update to patched version

**Summary:** 1 vulnerability ({severity})"""
    
    # Format for training
    text = f"""<s>[INST] {instruction}

{input_text} [/INST]

{response}</s>"""
    
    return {
        'text': text,
        'instruction': instruction,
        'input': input_text,
        'output': response,
        'metadata': {
            'cve_id': cve_id,
            'severity': severity,
            'ecosystem': ecosystem
        }
    }

def main():
    print("=" * 60)
    print("CVE to Training Dataset Converter (2024-2025)")
    print("=" * 60)
    
    # Find all CVE files from 2024 and 2025
    cve_files = []
    
    print("\n📂 Scanning for CVE files...")
    for year in ['2024', '2025']:
        year_path = f'/workspaces/ai_sec/datasets/sca/{year}'
        if Path(year_path).exists():
            files = glob.glob(f'{year_path}/**/*.json', recursive=True)
            cve_files.extend(files)
            print(f"  Found {len(files)} CVEs from {year}")
    
    print(f"\n📊 Total CVE files: {len(cve_files)}")
    
    # Process CVEs
    print("\n🔄 Converting CVEs to training format...")
    training_examples = []
    
    for i, cve_file in enumerate(cve_files):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(cve_files)} CVEs, created {len(training_examples)} training examples...")
        
        example = create_training_example(cve_file)
        if example:
            training_examples.append(example)
        
        # Limit to prevent too large dataset
        if len(training_examples) >= 10000:
            print(f"\n⚠️  Reached 10,000 training examples limit")
            break
    
    print(f"\n✅ Created {len(training_examples)} training examples")
    
    # Save dataset
    output_file = Path('/workspaces/ai_sec/datasets/sca_training_2024_2025.json')
    
    print(f"\n💾 Saving to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(training_examples, f, indent=2)
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"📁 File size: {file_size_mb:.2f} MB")
    
    # Show sample
    if training_examples:
        print(f"\n📝 Sample training example:")
        print(training_examples[0]['text'][:500] + "...")
    
    print("\n" + "=" * 60)
    print("✅ Conversion complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Push dataset to GitHub:")
    print("   git add datasets/sca_training_2024_2025.json")
    print("   git commit -m 'Add SCA training dataset'")
    print("   git push")
    print("\n2. Open Google Colab and run the training notebook")
    print("=" * 60)

if __name__ == "__main__":
    main()
