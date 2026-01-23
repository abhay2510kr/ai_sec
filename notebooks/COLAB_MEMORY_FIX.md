# Colab Memory Fix - Copy/Paste These Cells

## 🔴 BEFORE Step 5 (Load Model), ADD THIS NEW CELL:

```python
# Optimize CUDA memory allocation
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
print("✅ CUDA memory optimizations enabled")
```

---

## 🔴 IN Step 8 (Configure Training), REPLACE TrainingArguments with:

```python
from transformers import TrainingArguments, Trainer

# Check if GPU is available
use_fp16 = torch.cuda.is_available()

training_args = TrainingArguments(
    output_dir="/content/sca-package-checkpoints",
    num_train_epochs=2,  # Reduced from 3 to save memory
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,  # Increased from 16 (same learning, less memory)
    learning_rate=2e-4,
    fp16=use_fp16,
    save_strategy="epoch",  # Changed from steps
    logging_steps=10,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    max_grad_norm=0.3,
    gradient_checkpointing=True,  # CRITICAL: Saves ~30% GPU memory!
    report_to="none",
    save_total_limit=1,  # Only keep 1 checkpoint
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    # Removed eval_dataset to save memory
)

print("✅ Trainer configured!")
if use_fp16:
    print(f"\n⏱️  Estimated training time: 4-6 hours on T4 GPU (2 epochs)")
else:
    print(f"\n⏱️  Estimated training time: 24+ hours on CPU")
print(f"💾 Checkpoints will be saved to: /content/sca-package-checkpoints")
print(f"⚠️  Remember to download the model before session ends!")
print(f"\n🔧 Memory optimizations:")
print(f"   - Gradient checkpointing: ON")
print(f"   - Gradient accumulation: 32 steps")
print(f"   - Epochs: 2 (instead of 3)")
```

---

## 🔴 BEFORE Step 9 (Training), ADD THIS NEW CELL:

```python
# Clear GPU memory before training
import gc
import torch

torch.cuda.empty_cache()
gc.collect()

if torch.cuda.is_available():
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    total_mem = torch.cuda.mem_get_info()[1] / 1024**3
    print("🧹 GPU memory cleared")
    print(f"💾 Free GPU Memory: {free_mem:.2f} GB / {total_mem:.2f} GB")
    print(f"💾 Total GPU Memory: {total_mem:.2f} GB")
else:
    print("🧹 Memory cleared (CPU mode)")
```

---

## Summary of Changes:

✅ **Reduced epochs**: 3 → 2 (saves ~33% training time and memory)  
✅ **Gradient checkpointing**: ON (saves ~30% GPU memory)  
✅ **Gradient accumulation**: 16 → 32 (same effective learning, less memory)  
✅ **Save strategy**: Save only at epoch end (not every 100 steps)  
✅ **Checkpoint limit**: Keep only 1 checkpoint (saves disk space)  
✅ **Removed eval dataset**: No evaluation during training (saves memory)  
✅ **CUDA memory optimization**: Better memory fragmentation handling  
✅ **Memory clearing**: Clear cached memory before training starts  

These changes should reduce GPU memory usage from ~15GB to ~10-11GB, fitting comfortably in T4's 15GB!
