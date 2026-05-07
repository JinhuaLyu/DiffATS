"""
Kármán Vortex Street — Bulk Vorticity Data Generation
200 parameter sets × 50 clips × 200 frames = 10,000 clips total
Output: data/shard_NNN.pt, each a list of dicts with vor tensor + metadata
"""

import os
import sys
import time
import argparse
import itertools
import numpy as np
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import taichi as ti
import taichi.math as tm

try:
    ti.init(arch=ti.gpu)
    print("[Taichi] Using GPU (device 4)")
except Exception:
    ti.init(arch=ti.cpu)
    print("[Taichi] Fallback: Using CPU")


# ── Parameter grid (200 combinations) ────────────────────────────────────────
PARAM_GRID = list(itertools.product(
    [0.020, 0.022, 0.025, 0.028, 0.032],  # niu  → Re ≈ 80/73/64/57/50
    [26, 29, 32, 35, 38],                  # cx
    [56, 61, 67, 72],                      # cy
    [7, 9],                                # r
))  # 5×5×4×2 = 200


# ── LBM Solver ────────────────────────────────────────────────────────────────
@ti.data_oriented
class LBMSolver:
    def __init__(self, nx, ny, niu, bc_type, bc_value, cx, cy, r):
        self.nx, self.ny = nx, ny
        self.niu = niu
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau

        self.rho      = ti.field(float, shape=(nx, ny))
        self.vel      = ti.Vector.field(2, float, shape=(nx, ny))
        self.mask     = ti.field(float, shape=(nx, ny))
        self.f_old    = ti.Vector.field(9, float, shape=(nx, ny))
        self.f_new    = ti.Vector.field(9, float, shape=(nx, ny))
        self.vor_field = ti.field(float, shape=(nx, ny))

        self.w = ti.types.vector(9, float)(4,1,1,1,1,1/4,1/4,1/4,1/4) / 9.0
        self.e = ti.types.matrix(9, 2, int)(
            [0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]
        )
        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))

        self.cx = float(cx)
        self.cy = float(cy)
        self.r  = float(r)

    @ti.func
    def f_eq(self, i, j):
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.rho[i, j] * (1 + 3*eu + 4.5*eu*eu - 1.5*uv)

    @ti.kernel
    def init(self):
        self.vel.fill(0)
        self.rho.fill(1)
        self.mask.fill(0)
        for i, j in self.rho:
            self.f_old[i, j] = self.f_new[i, j] = self.f_eq(i, j)
            if (i - self.cx)**2 + (j - self.cy)**2 <= self.r**2:
                self.mask[i, j] = 1.0

    @ti.kernel
    def collide_and_stream(self):
        for i, j in ti.ndrange((1, self.nx-1), (1, self.ny-1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                feq = self.f_eq(ip, jp)
                self.f_new[i,j][k] = (1-self.inv_tau)*self.f_old[ip,jp][k] + feq[k]*self.inv_tau

    @ti.kernel
    def update_macro_var(self):
        for i, j in ti.ndrange((1, self.nx-1), (1, self.ny-1)):
            self.rho[i, j] = 0.0
            self.vel[i, j] = 0, 0
            for k in ti.static(range(9)):
                self.f_old[i,j][k] = self.f_new[i,j][k]
                self.rho[i,j] += self.f_new[i,j][k]
                self.vel[i,j] += tm.vec2(self.e[k,0], self.e[k,1]) * self.f_new[i,j][k]
            self.vel[i,j] /= self.rho[i,j]

    @ti.kernel
    def apply_bc(self):
        for j in range(1, self.ny-1):
            self.apply_bc_core(1, 0, 0,         j, 1,         j)
            self.apply_bc_core(1, 2, self.nx-1, j, self.nx-2, j)
        for i in range(self.nx):
            self.apply_bc_core(1, 1, i, self.ny-1, i, self.ny-2)
            self.apply_bc_core(1, 3, i, 0,         i, 1)
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.mask[i, j] == 1:
                self.vel[i, j] = 0, 0
                inb = i+1 if i >= self.cx else i-1
                jnb = j+1 if j >= self.cy else j-1
                self.apply_bc_core(0, 0, i, j, inb, jnb)

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        if outer == 1:
            if self.bc_type[dr] == 0:
                self.vel[ibc, jbc] = self.bc_value[dr]
            elif self.bc_type[dr] == 1:
                self.vel[ibc, jbc] = self.vel[inb, jnb]
        self.rho[ibc, jbc] = self.rho[inb, jnb]
        self.f_old[ibc, jbc] = self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]

    @ti.kernel
    def compute_vorticity(self):
        for i, j in ti.ndrange((1, self.nx-1), (1, self.ny-1)):
            dvdx = (self.vel[i+1,j][1] - self.vel[i-1,j][1]) * 0.5
            dudy = (self.vel[i,j+1][0] - self.vel[i,j-1][0]) * 0.5
            self.vor_field[i, j] = dvdx - dudy

    def step(self, n=1):
        for _ in range(n):
            self.collide_and_stream()
            self.update_macro_var()
            self.apply_bc()

    def get_vorticity(self):
        self.compute_vorticity()
        return self.vor_field.to_numpy()   # (nx, ny) float32


# ── Main ──────────────────────────────────────────────────────────────────────
def make_solver(niu, cx, cy, r):
    return LBMSolver(
        nx=128, ny=128,
        niu=niu,
        bc_type=[0, 0, 1, 0],             # left Dirichlet, right Neumann
        bc_value=[[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        cx=cx, cy=cy, r=r,
    )


def generate_param(param_idx, niu, cx, cy, r, args, dtype):
    T         = args.T
    n_clips   = args.clips
    sps       = args.steps_per_frame
    warmup    = args.warmup
    out_path  = os.path.join(args.out_dir, f"shard_{param_idx:03d}.pt")

    Re = round(0.1 * 2 * r / niu, 2)
    print(f"\n[Param {param_idx:03d}] niu={niu}, cx={cx}, cy={cy}, r={r}, Re={Re}")

    lbm = make_solver(niu, cx, cy, r)
    lbm.init()

    # Warmup
    lbm.step(warmup)
    step_counter = warmup

    buffer = []
    t0 = time.time()
    nan_reinit_count = 0

    for clip_idx in range(n_clips):
        frames = []
        step_start = step_counter
        ok = True

        for _ in range(T):
            lbm.step(sps)
            step_counter += sps
            vor = lbm.get_vorticity()

            if np.isnan(vor).any() or np.isinf(vor).any():
                print(f"  [WARN] NaN/Inf at clip {clip_idx}, frame {len(frames)} — reinitializing")
                nan_reinit_count += 1
                lbm.init()
                lbm.step(warmup)
                step_counter = warmup
                ok = False
                break

            frames.append(vor)

        if not ok:
            clip_idx -= 1  # retry this clip index
            continue

        vor_tensor = torch.tensor(
            np.stack(frames, axis=0).transpose(0, 2, 1),  # (T,128,128), y-axis up
            dtype=dtype
        )
        buffer.append({
            "vor":       vor_tensor,
            "param_idx": param_idx,
            "clip_idx":  clip_idx,
            "niu":       float(niu),
            "cx":        int(cx),
            "cy":        int(cy),
            "r":         int(r),
            "Re":        Re,
            "step_start": step_start,
        })

        if (clip_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  clip {clip_idx+1:3d}/{n_clips}  |  {elapsed:.1f}s elapsed")

    torch.save(buffer, out_path)
    print(f"  → Saved {len(buffer)} clips to {out_path}  (NaN reinits: {nan_reinit_count})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param-idx", type=int, default=None,
                        help="Only generate this param index (0-199).")
    parser.add_argument("--param-range", type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Generate param indices [START, END] inclusive. "
                             "Use to split 200 sets across parallel processes.")
    parser.add_argument("--T",     type=int, default=201, help="Frames per clip")
    parser.add_argument("--clips", type=int, default=50,  help="Clips per param set")
    parser.add_argument("--warmup",type=int, default=5000,help="Warmup steps before collecting")
    parser.add_argument("--steps-per-frame", type=int, default=5)
    parser.add_argument("--float16", action="store_true", help="Save as float16 (halves storage)")
    parser.add_argument("--out-dir", type=str, default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dtype = torch.float16 if args.float16 else torch.float32

    print(f"Total param sets: {len(PARAM_GRID)}")
    print(f"T={args.T}, clips={args.clips}, warmup={args.warmup}, sps={args.steps_per_frame}")
    print(f"dtype={dtype}, out_dir={args.out_dir}")

    if args.param_idx is not None:
        indices = [args.param_idx]
    elif args.param_range is not None:
        start, end = args.param_range
        indices = range(start, end + 1)
    else:
        indices = range(len(PARAM_GRID))

    for param_idx in indices:
        niu, cx, cy, r = PARAM_GRID[param_idx]
        out_path = os.path.join(args.out_dir, f"shard_{param_idx:03d}.pt")
        if os.path.exists(out_path):
            print(f"[Skip] shard_{param_idx:03d}.pt already exists")
            continue
        generate_param(param_idx, niu, cx, cy, r, args, dtype)

    print("\nAll done.")


if __name__ == "__main__":
    main()
