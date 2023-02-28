# %% 
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
    PLOTS = False
else:
    prefix = "../"  # or "../"
    PLOTS = True

# %% 
root_dir = Path(f"{prefix}data/clips_flume")
lset_dir = Path(f"{prefix}data/learning_sets/")
files = sorted(list(root_dir.glob("*_rgb.png")))
# %%
names = [f.name.split("_")[2] for f in files]
# %%
for f in files[:]: 
    cdir = lset_dir /  f.name.split("_")[2]
    if not cdir.is_dir():
        cdir.mkdir()

    shutil.copy(f, cdir / f.name)

# %%
