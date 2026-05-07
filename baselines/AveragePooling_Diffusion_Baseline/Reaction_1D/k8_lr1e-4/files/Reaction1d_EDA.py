import torch
import numpy as np
import matplotlib.pyplot as plt
import os

train_path = '/anvil/scratch/x-ezhou1/physics_datasets/data/reaction_1d/reaction_1d_train.pt'
test_path  = '/anvil/scratch/x-ezhou1/physics_datasets/data/reaction_1d/reaction_1d_test.pt'
save_dir   = '/anvil/scratch/x-ezhou1/physics_datasets/Experiments/Reaction_1d'
os.makedirs(save_dir, exist_ok=True)

train = torch.load(train_path, map_location='cpu', weights_only=False)
test  = torch.load(test_path,  map_location='cpu', weights_only=False)

for name, d in [('train', train), ('test', test)]:
    print(f'\n=== {name} ===')
    if isinstance(d, dict):
        for k, v in d.items():
            if hasattr(v, 'shape'):
                print(f'  {k}: shape={v.shape}, dtype={v.dtype}, '
                      f'min={v.min():.4f}, max={v.max():.4f}')
            else:
                print(f'  {k}: {type(v)}  -> {v}')
    elif isinstance(d, torch.Tensor):
        print(f'  shape={d.shape}, dtype={d.dtype}')
        print(f'  min={d.min():.4f}, max={d.max():.4f}')
    elif isinstance(d, (list, tuple)):
        print(f'  type=list, len={len(d)}')
        item = d[0]
        if isinstance(item, dict):
            for k, v in item.items():
                if hasattr(v, 'shape'):
                    print(f'    [{k}]: {v.shape}, {v.dtype}')
        elif hasattr(item, 'shape'):
            print(f'  item shape: {item.shape}')

print('\nEDA done!')
