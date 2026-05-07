"""
gt_norms.py — Report average ||v_gt||_1 and ||v_gt||_2 (Frobenius) over all
test samples for karman and burgers.
"""
import os
import sys
import numpy as np

sys.path.insert(0, '/home/x-jlyu5/jinhua/factor_diffusion/video')


def recon_karman(A, B, G):
    temp = np.einsum('txk,wk->txw', G, B, optimize=True)
    return np.einsum('ti,ixw->txw', A, temp, optimize=True)


def recon_burgers(A, B, G):
    temp = np.einsum('thk,wk->thw', G, B, optimize=True)
    return np.einsum('ti,ihw->thw', A, temp, optimize=True)


def run(exp):
    if exp == 'karman':
        sys.path.insert(0, '/home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/exp_karman_vortex/train')
        from dataset_karman_2d import KarmanTucker2DDataset as DS
        train = DS('/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/karman_vortex_2d/tucker_karman_rT10_rX128_rY30', split='all', device='cpu')
        test  = DS('/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/karman_vortex_2d/tucker_karman_rT10_rX128_rY30/test_data', split='all', device='cpu', external_stats=train.stats)
        A = test.UT_all.numpy().astype(np.float32)
        B = test.UY_all.numpy().astype(np.float32)
        G = test.G_all.numpy().astype(np.float32)
        recon = recon_karman
    else:
        sys.path.insert(0, '/home/x-jlyu5/jinhua/factor_diffusion/tensor_physics/exp_burgers_2d/train')
        from dataset_burgers_2d import BurgersTucker2DDataset as DS
        train = DS('/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/burgers_2d/tucker_burgers_rT5_rH20_rW20', split='all', device='cpu')
        test  = DS('/anvil/projects/x-eng260004/factor_diffusion/tucker_factors/burgers_2d/tucker_burgers_rT5_rH20_rW20/test_data', split='all', device='cpu', external_stats=train.stats)
        A = test.U1_all.numpy().astype(np.float32)
        B = test.U3_all.numpy().astype(np.float32)
        G = test.G_all.numpy().astype(np.float32)
        recon = recon_burgers

    N = A.shape[0]
    l1s = np.empty(N); l2s = np.empty(N); maxs = np.empty(N)
    for i in range(N):
        v = recon(A[i], B[i], G[i])
        l1s[i] = np.abs(v).sum()
        l2s[i] = np.linalg.norm(v)
        maxs[i] = np.abs(v).max()

    n_elements = 200 * 128 * 128   # (T*H*W)
    print(f'\n===== [{exp}]  N={N}  video_shape=(200,128,128)  n_elem={n_elements} =====')
    print(f'  ||v_gt||_1    mean={l1s.mean():.4e}  std={l1s.std():.4e}  '
          f'min={l1s.min():.4e}  max={l1s.max():.4e}')
    print(f'  ||v_gt||_2    mean={l2s.mean():.4e}  std={l2s.std():.4e}  '
          f'min={l2s.min():.4e}  max={l2s.max():.4e}')
    print(f'  max|v_gt|     mean={maxs.mean():.4e}  std={maxs.std():.4e}')
    print(f'  mean |v_gt|  (= L1/n_elem)  = {(l1s/n_elements).mean():.4e}')
    print(f'  RMS  v_gt    (= L2/sqrt(n)) = {(l2s/np.sqrt(n_elements)).mean():.4e}')


if __name__ == '__main__':
    for e in ('karman', 'burgers'):
        run(e)
