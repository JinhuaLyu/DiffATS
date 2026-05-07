import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse

class Burgers1dDataset(Dataset):
    def __init__(self, data_path, pool_k=4):
        self.pool_k = pool_k

        print(f"Loading: {data_path}")
        data = torch.load(data_path, map_location='cpu', weights_only=False)

        trajectories = data['tensor']   # [N, 201, 1024]
        nu           = data['nu']       # [N]

        N, T, L = trajectories.shape
        self.N    = N
        self.L_lr = L // pool_k        # 256 for pool_k=4

        print(f"Num trajectories : {N}")
        print(f"Spatial points   : {L} -> {self.L_lr} (pool_k={pool_k}, compression={pool_k}x)")
        print(f"Nu range         : [{nu.min():.4f}, {nu.max():.4f}]")
        print(f"Preloading and pooling into RAM...")

        # avg_pool1d: [N, 201, 1024] -> [N, 201, L_lr]
        traj_lr = F.avg_pool1d(trajectories, kernel_size=pool_k)

        self.cond_spatial = traj_lr[:, 0:1, :]   # [N, 1, L_lr]  t=0
        self.target       = traj_lr[:, 1:,  :]   # [N, 200, L_lr]  t=1..200
        self.cond_nu      = nu.unsqueeze(1)       # [N, 1]

        print(f"cond_spatial : {self.cond_spatial.shape}")
        print(f"cond_nu      : {self.cond_nu.shape}")
        print(f"target       : {self.target.shape}")
        print("Preload done!")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return {
            'cond_spatial': self.cond_spatial[idx],   # [1, L_lr]
            'cond_nu':      self.cond_nu[idx],        # [1]
            'target':       self.target[idx],          # [200, L_lr]
        }


# CLI test

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str,
        default='/scratch/bgxp/ezhou1/factor_diffusion_proj/data/burgers_1d/burgers_1d.pt')
    p.add_argument('--pool_k',      type=int, default=4)
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    ds = Burgers1dDataset(data_path=args.data_path, pool_k=args.pool_k)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers)
    batch = next(iter(loader))
    print(f"\ncond_spatial : {batch['cond_spatial'].shape}")   # [B, 1, 256]
    print(f"cond_nu      : {batch['cond_nu'].shape}")          # [B, 1]
    print(f"target       : {batch['target'].shape}")           # [B, 200, 256]
    print("Dataset ready.")
