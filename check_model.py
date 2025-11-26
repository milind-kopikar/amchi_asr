import safetensors
print('SafeTensors version:', safetensors.__version__)

from safetensors import safe_open
try:
    with safe_open('models/huggingface_konkani/checkpoint-5/model.safetensors', framework='pt') as f:
        keys = list(f.keys())
        print('Keys:', keys[:5], '...')  # Show first 5 keys
        print('Total keys:', len(keys))
except Exception as e:
    print('Error:', e)