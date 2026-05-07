import numpy as np
import pandas as pd

valid = np.load("valid_configs.npy", allow_pickle=True)
print(f"{len(valid)} valid configurations in total\n")

cds = sorted(set(float(c["convection_delta"]) for c in valid))
dgs = sorted(set(float(c["diffusion_gamma"])  for c in valid))
ics = sorted(set(str(c["ic_config"])          for c in valid))

print(f"valid convection_delta: {cds}")
print(f"valid diffusion_gamma:  {dgs}")
print(f"valid ic_config:        {ics}")

rows = [{"cd": float(c["convection_delta"]),
         "dg": float(c["diffusion_gamma"]),
         "l2": float(c["mean_l2"])} for c in valid]
df = pd.DataFrame(rows)
pivot = df.pivot_table(index="cd", columns="dg", values="l2", aggfunc="mean")
print("\nMean Rel-L2 (rows=cd, cols=dg, NaN=did not pass the constraint):")
print(pivot.to_string(float_format="{:.4f}".format))
