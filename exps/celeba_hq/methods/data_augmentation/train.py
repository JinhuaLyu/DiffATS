"""
Train data_augmentation = no_alignment data + per-batch random orthogonal
rotation (USigma -> USigmaQ, V -> VQ).

This is just a thin entry-point that runs no_alignment/train.main with
ortho_augment forced True (also propagated via the yaml in this dir).
"""

import os
import sys

# Run no_alignment's train.main with this dir's train.yaml.
_HERE = os.path.dirname(os.path.abspath(__file__))
_NO_ALIGN = os.path.normpath(os.path.join(_HERE, "..", "no_alignment"))
sys.path.insert(0, _NO_ALIGN)

# Override the default --config to point at our yaml unless the caller already
# passed --config explicitly.
if "--config" not in sys.argv:
    sys.argv.extend(["--config", os.path.join(_HERE, "train.yaml")])

# Execute no_alignment/train.py as if it were the entry-point.
import runpy  # noqa: E402
runpy.run_path(os.path.join(_NO_ALIGN, "train.py"), run_name="__main__")
