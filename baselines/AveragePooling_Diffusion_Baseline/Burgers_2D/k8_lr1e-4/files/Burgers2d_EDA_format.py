import torch

# shard
data = torch.load('/anvil/scratch/x-ezhou1/physics_datasets/data/burgers_2d/shard_00000.pt', 
                  map_location='cpu')
print(type(data))

# dict
if isinstance(data, dict):
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)} = {v}")

# list/tuple
elif isinstance(data, (list, tuple)):
    print(f"Length: {len(data)}")
    for i, item in enumerate(data[:3]):
        if hasattr(item, 'shape'):
            print(f"  [{i}]: shape={item.shape}, dtype={item.dtype}")

# tensor
elif hasattr(data, 'shape'):
    print(f"Tensor: shape={data.shape}, dtype={data.dtype}")
    print(f"  min={data.min():.4f}, max={data.max():.4f}")