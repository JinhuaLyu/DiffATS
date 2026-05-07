import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse

class Reaction1dDataset(Dataset):

    def __init__(self, data_path, pool_k=8):
        self.pool_k = pool_k

        print(f"Loading: {data_path}")
        data = torch.load(data_path, map_location='cpu', weights_only=False)

        trajectories = data['tensor']   # [N, 201, 1024]
        nu           = data['nu']       # [N]
        rho          = data['rho']      # [N]

        N, T, L = trajectories.shape
        self.N    = N
        self.L_lr = L // pool_k        # 128 for pool_k=8

        print(f"Num trajectories : {N}")
        print(f"Spatial points   : {L} -> {self.L_lr} (pool_k={pool_k}, compression={pool_k}x)")
        print(f"Nu  range        : [{nu.min():.6f}, {nu.max():.6f}]")
        print(f"Rho range        : [{rho.min():.4f}, {rho.max():.4f}]")
        print(f"Preloading and pooling into RAM...")

        # avg_pool1d: [N, 201, 1024] -> [N, 201, L_lr]
        traj_lr = F.avg_pool1d(trajectories, kernel_size=pool_k)

        self.cond_spatial = traj_lr[:, 0:1, :]          # [N, 1, L_lr]  t=0
        self.target       = traj_lr[:, 1:,  :]          # [N, 200, L_lr]  t=1..200
        # concat [nu, rho] -> [N, 2]
        self.cond_params  = torch.stack([nu, rho], dim=1)  # [N, 2]

        print(f"cond_spatial : {self.cond_spatial.shape}")
        print(f"cond_params  : {self.cond_params.shape}  ([nu, rho])")
        print(f"target       : {self.target.shape}")
        print("Preload done!")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return {
            'cond_spatial': self.cond_spatial[idx],   # [1, L_lr]
            'cond_params':  self.cond_params[idx],    # [2]
            'target':       self.target[idx],          # [200, L_lr]
        }

# CLI test

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str,
        default='/anvil/scratch/x-<user>/physics_datasets/data/reaction_1d/reaction_1d_train.pt')
    p.add_argument('--pool_k',      type=int, default=8)
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    ds = Reaction1dDataset(data_path=args.data_path, pool_k=args.pool_k)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers)
    batch = next(iter(loader))
    print(f"\ncond_spatial : {batch['cond_spatial'].shape}")  # [B, 1, 128]
    print(f"cond_params  : {batch['cond_params'].shape}")     # [B, 2]
    print(f"target       : {batch['target'].shape}")          # [B, 200, 128]
    print("Dataset ready.")
