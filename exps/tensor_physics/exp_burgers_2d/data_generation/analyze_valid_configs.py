import numpy as np
import pandas as pd

valid = np.load("valid_configs.npy", allow_pickle=True)
print(f"共 {len(valid)} 个有效配置\n")

cds = sorted(set(float(c["convection_delta"]) for c in valid))
dgs = sorted(set(float(c["diffusion_gamma"])  for c in valid))
ics = sorted(set(str(c["ic_config"])          for c in valid))

print(f"有效 convection_delta: {cds}")
print(f"有效 diffusion_gamma:  {dgs}")
print(f"有效 ic_config:        {ics}")

rows = [{"cd": float(c["convection_delta"]),
         "dg": float(c["diffusion_gamma"]),
         "l2": float(c["mean_l2"])} for c in valid]
df = pd.DataFrame(rows)
pivot = df.pivot_table(index="cd", columns="dg", values="l2", aggfunc="mean")
print("\n平均 Rel-L2（行=cd，列=dg，NaN=未通过约束）：")
print(pivot.to_string(float_format="{:.4f}".format))
