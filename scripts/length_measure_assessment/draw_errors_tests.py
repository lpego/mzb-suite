# %%
import argparse
import os, sys
import yaml

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pathlib import Path

from mzbsuite.utils import cfg_to_arguments, noneparse

# Set global configuration options
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

root = Path(".").resolve().parents[1]
root

# %%
# if __name__ == "__main__":
# parser = argparse.ArgumentParser()
parser = {}
parser["config_file"] = root / "configs/configuration_flume_datasets.yaml"
parser["model_annotations"] = (
    root
    / "results/project_portable_flume/skeletons/supervised_skeletons/skseg_mit-b2-vJuly_flume_20230703_2019/size_skel_supervised_model.csv"
)
parser["manual_annotations"] = (
    root
    / "data/learning_sets/project_portable_flume/skeletonization/manual_annotations_summary.csv"
)
parser["verbose"] = True

# parser.add_argument("--config_file", type=str, required=True)
# parser.add_argument("--input_dir", type=str, required=True)
# parser.add_argument("--list_of_files", type=noneparse, required=False, default=None)
# parser.add_argument("--output_dir", type=str, required=True)
# parser.add_argument("--save_masks", type=str, required=True)
# parser.add_argument("--verbose", "-v", action="store_true")
# args = parser.parse_args()

args = cfg_to_arguments(parser)
args.input_dir = Path(args.input_dir)
# args.output_dir = Path(args.output_dir)

with open(args.config_file, "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
cfg = cfg_to_arguments(cfg)


# sys.exit(main(args, cfg))

# %%
manual_annotations = pd.read_csv(args.manual_annotations, index_col=False)
if "Unnamed: 0" in manual_annotations.columns:
    manual_annotations = manual_annotations.drop(columns=["Unnamed: 0"])

manual_annotations["clip_name"] = [
    "_".join(f.split("_")[:-1]) for f in manual_annotations["clip_name"]
]
manual_annotations = manual_annotations.set_index("clip_name", drop=True)
# %%
auto_annotations = pd.read_csv(args.model_annotations, index_col="clip_name")

# %%

# auto_annotations["clip_name"] = [
#     f.split("/")[-1].split(".")[0] for f in auto_annotations.clip_filename
# ]
# auto_annotations["clip_name"] = [
#     "_".join(f.split("_")[:-1]) for f in auto_annotations["clip_name"]
# ]
# auto_annotations = auto_annotations.set_index("clip_name", drop=True)

# %%
merged_annotations = manual_annotations.merge(
    auto_annotations, left_index=True, right_index=True, how="inner"
)
# %%
merged_annotations["abs_error_bodysize_skel"] = (
    merged_annotations["body_length"] - merged_annotations["skel_length"]
)

try:
    merged_annotations["ellipse_major"]
    is_ell = True
except:
    is_ell = False

if is_ell:
    merged_annotations["abs_error_bodysize_ell"] = (
        merged_annotations["body_length"] - merged_annotations["ellipse_major"]
    )
    merged_annotations["abs_error_headsize"] = (
        merged_annotations["head_length"] - merged_annotations["ellipse_minor"]
    )
# %%
plt.figure()
merged_annotations.groupby("species").mean()[
    ["body_length", "abs_error_bodysize_skel"]
].plot(kind="bar", rot=90)

if is_ell:
    plt.figure()
    merged_annotations.groupby("species").mean()[
        ["body_length", "abs_error_bodysize_ell"]
    ].plot(kind="bar", rot=90)

    plt.figure()
    merged_annotations.groupby("species").mean()[
        ["head_length", "abs_error_bodysize_ell"]
    ].plot(kind="bar", rot=90)

# %%
merged_annotations["rel_error_bodysize_skel"] = (
    100
    * np.abs(merged_annotations["body_length"] - merged_annotations["skel_length"])
    / merged_annotations["body_length"]
)

if is_ell:
    merged_annotations["rel_error_bodysize_ell"] = (
        100
        * np.abs(
            merged_annotations["body_length"] - merged_annotations["ellipse_major"]
        )
        / merged_annotations["body_length"]
    )
    merged_annotations["rel_error_headsize"] = (
        100
        * np.abs(
            merged_annotations["head_length"] - merged_annotations["ellipse_minor"]
        )
        / merged_annotations["body_length"]
    )
# %%
plt.figure()
merged_annotations.groupby("species").mean()[["rel_error_bodysize_skel"]].plot(
    kind="bar", rot=90
)

if is_ell:
    plt.figure()
    merged_annotations.groupby("species").mean()[["rel_error_bodysize_ell"]].plot(
        kind="bar", rot=90
    )

    plt.figure()
    merged_annotations.groupby("species").mean()[["rel_error_bodysize_ell"]].plot(
        kind="bar", rot=90
    )

# %%
# merged_annotations.to_csv("../../../mzb-classification/merged_area_based_v2.csv")
# %%

areas = np.array([0, 20000, 50000, 150000, 200000, np.Inf])
topl = merged_annotations.groupby(pd.cut(merged_annotations["area"], areas)).mean()

# %%
plt.figure()
topl[["rel_error_bodysize_skel"]].plot(kind="bar", rot=90)

plt.figure()
merged_annotations.groupby("species").mean()[
    ["body_length", "rel_error_bodysize_skel"]
].plot(kind="bar", rot=90)

# plt.figure()
# merged_annotations.groupby("species").mean()[["area", "abs_error_bodysize_ell"]].plot(kind="bar", rot=90)

# plt.figure()
# merged_annotations.groupby("species").mean()[["area", "abs_error_bodysize_ell"]].plot(kind="bar", rot=90)

# df.groupby(pd.cut(df["B"], np.arange(0, 1.0+0.155, 0.155))).sum(
# %%
merged_annotations.groupby("species").mean()[["area"]].sort_values(by="area").plot(
    kind="bar", rot=90
)

# %%
