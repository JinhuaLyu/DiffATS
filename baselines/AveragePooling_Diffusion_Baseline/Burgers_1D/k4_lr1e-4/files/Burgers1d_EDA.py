import torch
import numpy as np
import matplotlib.pyplot as plt
import os

train_path = '${DATA_ROOT}/data/burgers_1d/burgers_1d.pt'
test_path  = '${DATA_ROOT}/data/burgers_1d/burgers_1d_test.pt'
save_dir   = '${DATA_ROOT}/Experiments/Burgers_1d'
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
                print(f'  {k}: {type(v)}')
    elif isinstance(d, torch.Tensor):
        print(f'  shape={d.shape}, dtype={d.dtype}')
        print(f'  min={d.min():.4f}, max={d.max():.4f}')
    elif isinstance(d, (list, tuple)):
        print(f'  type=list/tuple, len={len(d)}')
        item = d[0]
        if isinstance(item, dict):
            for k, v in item.items():
                if hasattr(v, 'shape'):
                    print(f'    [{k}]: {v.shape}, {v.dtype}')
        elif hasattr(item, 'shape'):
            print(f'  item shape: {item.shape}')

# 3 random training samples
if isinstance(train, torch.Tensor) and train.dim() >= 2:
    N = train.shape[0]
    idx = torch.randperm(N)[:3]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    for i, ax in enumerate(axes):
        sample = train[idx[i]]
        if sample.dim() == 2:
            # [T, L] -> plot a few timesteps
            for t in [0, sample.shape[0]//4,
                      sample.shape[0]//2, -1]:
                ax.plot(sample[t].numpy(), label=f't={t}')
            ax.legend(fontsize=8)
        else:
            ax.plot(sample.numpy())
        ax.set_title(f'Sample {idx[i].item()}')
        ax.set_xlabel('x')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eda_train_samples.png'), dpi=120)
    plt.close()
    print(f'\nPlot saved: {save_dir}/eda_train_samples.png')

print('EDA done!')

