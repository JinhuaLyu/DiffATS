import torch

d = torch.load('${DATA_ROOT}/data/moving_mnist/moving_mnist_20k_2slow.pt', map_location='cpu')
print('Type:', type(d))
if isinstance(d, dict):
    for k, v in d.items():
        if hasattr(v, 'shape'):
            print(f'  {k}: {v.shape}, {v.dtype}')
        else:
            print(f'  {k}: {type(v)}')

elif isinstance(d, (list, tuple)):
    print('Length:', len(d))
    item = d[0]
    if hasattr(item, 'shape'):
        print('Item shape:', item.shape, item.dtype)
    elif isinstance(item, dict):
        for k, v in item.items():
            if hasattr(v, 'shape'):
                print(f'  {k}: {v.shape}')
                
elif hasattr(d, 'shape'):
    print('Shape:', d.shape, d.dtype)
    print('Min:', d.min().item(), 'Max:', d.max().item())
