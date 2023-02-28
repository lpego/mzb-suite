# %%
%load_ext autoreload
%autoreload 2 

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Set global configuration options
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7


import os

os.chdir("/data/shared/swiss-invertebrates-data/scripts/length_measure_assessment")

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
    PLOTS = False
else:
    prefix = "../../"  # or "../"
    PLOTS = True


# %% 
manual_annotations = pd.read_csv(f"{prefix}data/2021_swiss_invertebrates/manual_annotations_summary.csv")
manual_annotations["clip_name"] = ["_".join(f.split("_")[:-1]) for f in manual_annotations["clip_name"]]
manual_annotations = manual_annotations.set_index("clip_name", drop=True)
manual_annotations = manual_annotations.drop(columns=["Unnamed: 0"])

# auto_annotations = pd.read_csv(f"{prefix}/FLUME_MASK_ATTRIBUTES_AREA_BASED_PARS.csv")
auto_annotations = pd.read_csv(f"{prefix}/FLUME_MASK_ATT_MYSKEL.csv")
auto_annotations["clip_name"] = [f.split("/")[-1].split(".")[0] for f in auto_annotations.clip_filename]
auto_annotations["clip_name"] = ["_".join(f.split("_")[:-1]) for f in auto_annotations["clip_name"]]
auto_annotations = auto_annotations.set_index("clip_name", drop=True)
auto_annotations = auto_annotations.drop(columns=["Unnamed: 0"])

# %%
merged_annotations = manual_annotations.merge(auto_annotations, left_index=True, right_index=True, how="inner")
merged_annotations["abs_error_bodysize_skel"] = merged_annotations["body_length"] - merged_annotations["skel_length"]

try: 
    merged_annotations["ellipse_major"]
    is_ell = True
except: 
    is_ell = False

if is_ell:
    merged_annotations["abs_error_bodysize_ell"] = merged_annotations["body_length"] - merged_annotations["ellipse_major"]
    merged_annotations["abs_error_headsize"] = merged_annotations["head_length"] - merged_annotations["ellipse_minor"]
# %%
plt.figure()
merged_annotations.groupby("species").mean()[["body_length", "abs_error_bodysize_skel"]].plot(kind="bar", rot=90)

if is_ell:    
    plt.figure()
    merged_annotations.groupby("species").mean()[["body_length", "abs_error_bodysize_ell"]].plot(kind="bar", rot=90)

    plt.figure()
    merged_annotations.groupby("species").mean()[["head_length", "abs_error_bodysize_ell"]].plot(kind="bar", rot=90)

# %%
merged_annotations["rel_error_bodysize_skel"] = 100*np.abs(merged_annotations["body_length"] - merged_annotations["skel_length"]) / merged_annotations["body_length"] 

if is_ell:
    merged_annotations["rel_error_bodysize_ell"] = 100*np.abs(merged_annotations["body_length"] - merged_annotations["ellipse_major"]) / merged_annotations["body_length"]
    merged_annotations["rel_error_headsize"] = 100*np.abs(merged_annotations["head_length"] - merged_annotations["ellipse_minor"]) / merged_annotations["body_length"]
# %%
plt.figure()
merged_annotations.groupby("species").mean()[["rel_error_bodysize_skel"]].plot(kind="bar", rot=90)

if is_ell:
    plt.figure()
    merged_annotations.groupby("species").mean()[["rel_error_bodysize_ell"]].plot(kind="bar", rot=90)

    plt.figure()
    merged_annotations.groupby("species").mean()[["rel_error_bodysize_ell"]].plot(kind="bar", rot=90)

# %%
# merged_annotations.to_csv("../../../mzb-classification/merged_area_based_v2.csv")
# %%

areas = np.array([0, 20000, 50000, 150000, 200000, np.Inf])
topl = merged_annotations.groupby(pd.cut(merged_annotations["area"], areas)).mean()

# %% 
plt.figure()
topl[["rel_error_bodysize_skel"]].plot(kind="bar", rot=90)

plt.figure()
merged_annotations.groupby("species").mean()[["body_length", "rel_error_bodysize_skel"]].plot(kind="bar", rot=90)

# plt.figure()
# merged_annotations.groupby("species").mean()[["area", "abs_error_bodysize_ell"]].plot(kind="bar", rot=90)

# plt.figure()
# merged_annotations.groupby("species").mean()[["area", "abs_error_bodysize_ell"]].plot(kind="bar", rot=90)

# df.groupby(pd.cut(df["B"], np.arange(0, 1.0+0.155, 0.155))).sum(
# %%
merged_annotations.groupby("species").mean()[["area"]].sort_values(by="area").plot(kind="bar", rot=90)

# %%
