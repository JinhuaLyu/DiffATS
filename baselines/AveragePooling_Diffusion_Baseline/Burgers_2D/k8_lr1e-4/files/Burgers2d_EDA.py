import torch
import os

test_dir = '/anvil/scratch/x-<user>/physics_datasets/data/burgers_2d/test_data'
test_files = [f for f in os.listdir(test_dir) if f.endswith('.pt')]
print("Test files:", test_files)

test_data = torch.load(os.path.join(test_dir, test_files[0]), map_location='cpu')
print("Type:", type(test_data))
print("Length:", len(test_data))

# first sample
sample = test_data[0]
print("\nSample type:", type(sample))

if isinstance(sample, dict):
    for k, v in sample.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, min={v.min():.4f}, max={v.max():.4f}")
        else:
            print(f"  {k}: {v}")
elif isinstance(sample, (list, tuple)):
    for i, item in enumerate(sample):
        if hasattr(item, 'shape'):
            print(f"  [{i}]: shape={item.shape}, dtype={item.dtype}")
elif hasattr(sample, 'shape'):
    print(f"Tensor: shape={sample.shape}, dtype={sample.dtype}")
    print(f"  min={sample.min():.4f}, max={sample.max():.4f}")
