# 🚀 Quick Start Guide - Train Your First AI Security Model

## Step-by-Step Instructions

### **Step 1: Download CVE Data (10-15 minutes)**

```bash
# Run in terminal
cd /workspaces/ai_sec
python scripts/01_download_nvd_simple.py
```

**What this does:**
- Downloads ~40,000 recent CVEs (2023-2024) from NVD
- Saves to: `datasets/sca/nvd_cves_recent.json`
- Takes 10-15 minutes due to API rate limits

---

### **Step 2: Create Training Dataset (1-2 minutes)**

```bash
# Run in terminal
python scripts/02_create_sca_dataset.py
```

**What this does:**
- Converts CVE data into training examples
- Creates 1,000 instruction-response pairs
- Saves to: `datasets/sca/sca_training_dataset.json`

---

### **Step 3: Upload to Google Drive**

1. Download `datasets/sca/sca_training_dataset.json` from VS Code
2. Go to [Google Drive](https://drive.google.com)
3. Create folder: `ai_sec`
4. Upload `sca_training_dataset.json` to that folder

---

### **Step 4: Open Google Colab**

1. Go to: https://colab.research.google.com/
2. Click: **File → Upload notebook**
3. Upload: `notebooks/train_sca_package_colab.ipynb`
4. Or click: **File → Open notebook → GitHub**
   - Enter your repo: `abhay2510kr/ai_sec`
   - Select: `notebooks/train_sca_package_colab.ipynb`

---

### **Step 5: Enable GPU in Colab**

1. In Colab, click: **Runtime → Change runtime type**
2. Select: **T4 GPU**
3. Click: **Save**

---

### **Step 6: Run Training (2-4 hours)**

1. **Run each cell** in the notebook with `Shift+Enter`
2. **Mount Google Drive** when prompted
3. **Wait for training** (2-4 hours)
   - You can close the tab - training continues!
   - Check progress by reopening the notebook

---

### **Step 7: Test Your Model**

Run the final cell to test:

```python
# Test input
test_input = """[INST] Analyze this package.json for vulnerabilities

```json
{
  "dependencies": {
    "express": "4.16.0",
    "lodash": "4.17.4"
  }
}
``` [/INST]"""

# Model will respond with vulnerabilities!
```

---

## 🎉 Success!

You've trained your first AI security model!

**What you have now:**
- ✅ Trained SCA model in Google Drive
- ✅ Can detect package vulnerabilities
- ✅ Ready to use in production

**Next Steps:**
1. Train more models (SAST, IaC, Container)
2. Deploy using vLLM (see `docs/04_MODEL_DEPLOYMENT.md`)
3. Integrate into CI/CD (see `docs/05_INTEGRATION_ORCHESTRATION.md`)

---

## 💡 Tips

**If Colab disconnects:**
- Training checkpoints are saved every 50 steps
- Just re-run the training cell - it will resume!

**Want faster training?**
- Use Colab Pro ($10/month) for A100 GPU (4x faster)
- Or use Vast.ai RTX 4090 ($0.50/hour, 2x faster)

**Need help?**
- Check: `docs/06_BUDGET_TRAINING_OPTIONS.md`
- All documentation is in the `docs/` folder

---

## 📊 Cost Summary

| Option | Cost | Time |
|--------|------|------|
| **Google Colab Free** | $0 | 2-4 hours |
| **Colab Pro** | $10/month | 30-60 min |
| **Vast.ai RTX 4090** | $2-3 | 1-2 hours |

**Recommended:** Start FREE with Colab, upgrade later if needed!
