# GPU vs CPU: Running Konkani ASR on Your Laptop

## The CUDA Question: Do You Need NVIDIA GPU?

**Short Answer**: ❌ **No, you don't need NVIDIA GPU chips!** The code will run on CPU.

## How GPU/CPU Detection Works

### Automatic Device Selection

The code automatically detects and uses the best available hardware:

```python
import torch

# Check what's available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use NVIDIA GPU
    print("🎮 Using NVIDIA GPU (fast)")
else:
    device = torch.device("cpu")   # Use CPU
    print("💻 Using CPU (slower but works)")
```

### Your Laptop Reality

**✅ CPU-Only Operation**: Your laptop will automatically use CPU mode.

**⚠️ Performance Impact**: Training will be **10-20x slower** than GPU, but it works!

## Performance Comparison

### Training Times (Estimated)

| Hardware | 1 Epoch | 10 Epochs | Status |
|----------|---------|-----------|--------|
| **NVIDIA RTX 3080** | 5 minutes | 50 minutes | ⚡ Fast |
| **NVIDIA GTX 1650** | 15 minutes | 2.5 hours | 🚀 Good |
| **Your Laptop CPU** | 1-2 hours | 10-20 hours | ✅ Works |

### Memory Requirements

| Component | GPU Memory | CPU Memory |
|-----------|------------|------------|
| IndicConformer Model | ~2GB VRAM | ~4GB RAM |
| Training Batch | ~1GB VRAM | ~2GB RAM |
| **Total Minimum** | **4GB VRAM** | **8GB RAM** |

## Will Your Laptop Work?

### ✅ Requirements Check

**Minimum Specs for CPU Training:**
- **RAM**: 16GB (recommended)
- **CPU**: Modern multi-core (i5/i7/i9 or equivalent)
- **Storage**: 50GB free space
- **OS**: Windows/Linux/Mac

**Your Current Setup:**
- Python 3.13.7 ✅
- Windows 11 ✅
- Should work on most modern laptops

### ⚠️ Potential Issues

1. **Memory**: If <16GB RAM, might need smaller batches
2. **Time**: Training takes much longer on CPU
3. **Heat**: CPU training generates heat

## Testing Strategy for Your Laptop

### Phase 1: Quick CPU Test (What We Do First)

**✅ Perfect for your laptop!**

```bash
# Test with minimal data (30 seconds)
python scripts/minimal_test.py \
  --audio_file your_audio.wav \
  --transcript "your text"

# This will:
# - Run on CPU automatically
# - Take 5-15 minutes total
# - Verify everything works
```

### Phase 2: Full Training (Optional)

**⚠️ Will take 10-20 hours on CPU**

```bash
# Full training with your 10 minutes
python scripts/fine_tune.py --config configs/konkani_finetune.yaml
```

## Optimizing for CPU Training

### Configuration Adjustments

For CPU training, modify `configs/konkani_finetune.yaml`:

```yaml
trainer:
  devices: 0          # Use CPU (0 = CPU, 1+ = GPU count)
  max_epochs: 5       # Fewer epochs for testing
  accumulate_grad_batches: 8  # Larger batches for CPU efficiency

model:
  train_ds:
    batch_size: 2     # Smaller batches for CPU memory
  val_ds:
    batch_size: 2
```

### Memory Optimization

```python
# In training scripts, add:
if not torch.cuda.is_available():
    # CPU-specific optimizations
    torch.set_num_threads(4)  # Use multiple CPU cores
    os.environ['OMP_NUM_THREADS'] = '4'
```

## Alternative: Cloud GPU (If Needed)

If CPU training is too slow, you can use cloud GPUs later:

### Free/Low-Cost Options:
- **Google Colab**: Free GPU (Tesla T4)
- **Kaggle**: Free GPU instances
- **AWS SageMaker**: Pay-per-use GPU
- **Paperspace**: GPU cloud instances

### Colab Example:
```python
# Run this in Colab notebook:
!pip install -r requirements.txt
!python scripts/minimal_test.py --audio_file audio.wav --transcript "text"
```

## Your Next Steps

### ✅ Immediate (CPU on Your Laptop)

1. **Test minimal training** with your 10 minutes of audio
2. **Verify CPU mode works** (should auto-detect)
3. **Measure time** for one training epoch

### 🚀 Later (If Needed)

1. **Consider cloud GPU** if CPU is too slow
2. **Optimize batch sizes** for your RAM
3. **Use fewer epochs** for testing

## Summary

**🎯 Your laptop will work perfectly for testing!**

- **CUDA/GPU**: Not required - code auto-detects CPU
- **Performance**: Slower but functional
- **Testing**: Start with minimal test (5-15 minutes)
- **Full Training**: Possible but time-consuming (10-20 hours)

**Ready to test the manifest generator with your 10 minutes of audio on CPU?** The minimal test will confirm everything works before committing to longer training! 🚀

*Note: Most ASR research actually happens on CPU during development - GPUs are for production training.*</content>
<parameter name="filePath">C:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\CPU_GPU_GUIDE.md