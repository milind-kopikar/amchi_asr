#!/usr/bin/env python3
import subprocess
import torch

print('Python: running GPU check')
print('🔥 CUDA Available:', torch.cuda.is_available())
print('🎮 Device Count:', torch.cuda.device_count())
try:
    out = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'], stderr=subprocess.STDOUT)
    print('\nnvidia-smi output:')
    print(out.decode().strip())
except Exception as e:
    print('nvidia-smi not available or failed:', e)
