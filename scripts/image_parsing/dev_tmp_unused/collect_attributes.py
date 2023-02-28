# %%
from pathlib import Path

import cv2
# import yaml
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# import pathlib


# import imutils

# from skimage import measure
# import skimage
# import skimage.segmentation

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
else:
    prefix = "../"  # or "../"
# %%

main_root = Path(
    f"{prefix}data/data_raw_custom_processing/project_portable_flume/"
)
plots_out = main_root / "plots"
plots_out.mkdir(exist_ok=True, parents=True)

conv_rate = np.mean([133.1, 136.6, 133.2, 133.2, 133.2, 118.6, 133.4, 132.0 ]) # px / mm
sites = list(main_root.glob("*"))
sites = [a for a in sites if "plots" not in str(a)]

data_summary = []
for site in sites[:]: 
    experiments = list(site.glob("*"))
    for exp in experiments[:]: 
        files_proc = list(exp.glob("**/*.csv"))
        for i, fi in enumerate(files_proc[:]):
            print(f"{i+1}/{len(files_proc)}: {site.name}, {exp.name}, {fi.name}")
            atts_ = pd.read_csv(fi).drop(columns=["Unnamed: 0"])
            atts_["site"] = site.name
            atts_["experiment"] = exp.name
            atts_["conversion"] = conv_rate
            atts_["area_ratio"] = atts_.bbox_area / atts_.area_px

            data_summary.append(atts_)

data_summary = pd.concat(data_summary).reset_index().drop(columns=["index"])
# classification choices
ind_s = pd.concat(
    (
        data_summary.area_px < 5000,
        # summary.h < 50,
        # summary.v < 100,
        # summary.average_color_std > 65,
        # (summary.area_ratio < 1.5) | (summary.area_ratio > 10),
    ),
    axis=1,
)
data_summary = data_summary.drop(index=np.where(np.any(ind_s, axis=1))[0]).copy()
data_summary = data_summary.reset_index(drop=True)
# data_summary["exp_id"] = data_summary["site"] + "_" + data_summary["experiment"]
# data_summary["exp_id"]
# %%
for site in data_summary.site.unique():
    sub_d = data_summary[data_summary.site == site]
    N_e = sub_d.groupby("experiment").count().input_file
    F_e = sub_d.groupby("experiment").first()

    plt.figure(figsize=(15, 6))
    plt.title(f'{" ".join([a.capitalize() for a in F_e.site.iloc[0].replace("_", " ").split()])}: Counts')
    plt.bar(range(N_e.shape[0]), N_e)
    plt.grid("on")
    plt.ylabel("N detected")
    plt.xlabel("Site Experiment")
    plt.xticks(range(N_e.shape[0]), [a.replace("_", " ").capitalize() for a in F_e.index.tolist()], rotation=90)
    plt.tight_layout()
    plt.savefig(plots_out / f"{F_e.site.iloc[0]}_counts_per_exp.pdf")
    plt.show()

for site in data_summary.site.unique():
    sub_d = data_summary[data_summary.site == site]
    sq_mm_area = sub_d.area_px * 1/(conv_rate**2)
    # mm^2 = px^2 * 1/(px/mm)^2 = px^2 * mm^2 / px^2 = mm^2  

    bbii = np.arange(0, 21, 1)
    sq_mm_area = np.clip(sq_mm_area, bbii[0], bbii[-1])
    plt.figure(figsize=(15, 6))
    co, bins, plo = plt.hist(sq_mm_area, bins=bbii, histtype="bar", align="mid")
    plt.grid("yaxis")

    xlabels = [f"{bbii[i]}-{bbii[i+1]}" for i in range(len(bbii)-1)]
    bbii[:].astype(str)
    xlabels[-1] += '>'
    plt.xticks(bbii[:-1]+0.5, xlabels, rotation=90)
    plt.ylabel("Counts")
    plt.xlabel("Area class [mm^2]")
    plt.tight_layout()
    plt.savefig(plots_out / f"{site}_bin_class_counts_per_exp.pdf")
    plt.show()
# %%
for site in data_summary.site.unique():
    sub_d = data_summary[data_summary.site == site]
    for exp in sub_d.experiment.unique():
        subsub_d = sub_d[sub_d.experiment == exp]
        sq_mm_area = subsub_d.area_px * 1/(conv_rate**2)
        # mm^2 = px^2 * 1/(px/mm)^2 = px^2 * mm^2 / px^2 = mm^2  

        bbii = np.arange(0, 21, 1)
        sq_mm_area = np.clip(sq_mm_area, bbii[0], bbii[-1])
        plt.figure(figsize=(15, 6))
        co, bins, plo = plt.hist(sq_mm_area, bins=bbii, histtype="bar", align="mid")
        plt.grid("yaxis")

        xlabels = [f"{bbii[i]}-{bbii[i+1]}" for i in range(len(bbii)-1)]
        bbii[:].astype(str)
        xlabels[-1] += '>'
        plt.xticks(bbii[:-1]+0.5, xlabels, rotation=90)
        plt.ylabel("Counts")
        plt.xlabel("Area class [mm^2]")
        plt.tight_layout()
        plt.savefig(plots_out / f"{site}_{exp}_bin_class_counts_per_exp.pdf")
        plt.show()
# %%

# %%
