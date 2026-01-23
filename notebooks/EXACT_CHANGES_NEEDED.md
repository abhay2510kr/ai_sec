# 🔧 COLAB FIXES - Exact Changes Needed

## ⚠️ If you're running the OLD notebook in Colab, make these changes:

---

## 📍 CHANGE 1: In Step 6 (Configure LoRA)

**FIND this cell:**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

**REPLACE with:**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# CRITICAL FIX: Enable gradients for LoRA
model.enable_input_require_grads()
print("✅ Enabled input gradients for LoRA")

model.print_trainable_parameters()
```

---

## 📍 CHANGE 2: In Step 8 (Configure Training)

**FIND this section:**
```python
training_args = TrainingArguments(
    output_dir="/content/sca-package-checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    # ... other settings ...
    gradient_checkpointing=True,  # ← REMOVE THIS LINE
)
```

**REPLACE with:**
```python
training_args = TrainingArguments(
    output_dir="/content/sca-package-checkpoints",
    num_train_epochs=2,  # Reduced to 2
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,  # Increased to 32
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    max_grad_norm=0.3,
    report_to="none",
    save_total_limit=1,
    # DO NOT add gradient_checkpointing=True (incompatible with LoRA)
)
```

---

## 📍 CHANGE 3: Before Step 9 (Training)

**ADD A NEW CELL:**
```python
# Clear GPU memory before training
import gc
import torch

torch.cuda.empty_cache()
gc.collect()

if torch.cuda.is_available():
    free = torch.cuda.mem_get_info()[0] / 1024**3
    total = torch.cuda.mem_get_info()[1] / 1024**3
    print(f"🧹 GPU memory cleared")
    print(f"💾 Free: {free:.2f} GB / {total:.2f} GB")
```

---

## ✅ OR... Just Use the Updated Notebook!

**Download the already-fixed notebook:**
`notebooks/train_sca_package_colab_optimized.ipynb`

Upload it to Colab and run - all fixes are already applied! 🎉

---

## 📋 Summary of Fixes:

1. ✅ Added `model.enable_input_require_grads()` after LoRA
2. ✅ Removed `gradient_checkpointing=True` from TrainingArguments
3. ✅ Reduced epochs: 3 → 2
4. ✅ Increased gradient accumulation: 16 → 32
5. ✅ Added memory clearing before training
6. ✅ Changed save strategy to "epoch"
7. ✅ Limited checkpoints to 1

These changes fix the gradient error and reduce memory usage!
