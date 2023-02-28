
# %%
%load_ext autoreload
%autoreload 2 

import os
import shutil

import cv2
import numpy as np
import pandas as pd

print(os.getcwd())

from pathlib import Path

from matplotlib import pyplot as plt

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
    PLOTS = False
else:
    prefix = "../"  # or "../"
    PLOTS = True
# %% 
# Get clip based on the fact that it is an existing annotation in Daninas folder
annot_dir = Path(f"{prefix}/data/2021_swiss_invertebrates/manual_measurements/")
annot_files = sorted(list(annot_dir.glob("**/*/line_V4.csv")))

store_files = Path(f"{prefix}/data/skel_segm_thin")
(store_files / "images").mkdir(exist_ok=True, parents=True)
(store_files / "sk_body").mkdir(exist_ok=True, parents=True)
(store_files / "sk_head").mkdir(exist_ok=True, parents=True)

THICKNESS=1
# %% 
clips_root  = Path(f"{prefix}/data/data_raw_custom_processing/project_portable_flume/clips_5k/")
PLT = False
for file in annot_files[:]:
    
    gen_name = "_".join(file.parent.name.split("__")[1].split("_")[:-1])
    rgb_clip = gen_name + "_rgb.png"

    # 1 copy actual existing clip. 
    shutil.copy(clips_root / rgb_clip, store_files / "images" )

    test_f_im = Path(clips_root / rgb_clip)
    test_im = cv2.cvtColor(cv2.imread(str(test_f_im)), cv2.COLOR_BGR2RGB)
    
    if PLT: 
        plt.figure()
        plt.imshow(test_im)

    line = pd.read_csv(file)
    body = line[["x_coords", "y_coords"]].loc[line.annotation_id == "body"].values
    head = line[["x_coords", "y_coords"]].loc[line.annotation_id == "head"].values

    # Draw the polyline on the image using cv2.polylines()
    bw_mask = np.zeros_like(test_im)
    body = cv2.polylines(bw_mask, [body], isClosed=False, color=(255, 255, 255), thickness=THICKNESS)
    cv2.imwrite(str(store_files / "sk_body" / f"{gen_name}_body_skel.png"), body)
    
    if PLT: 
        plt.figure()
        plt.imshow(body)

    bw_mask = np.zeros_like(test_im)
    head = cv2.polylines(bw_mask, [head], isClosed=False, color=(255, 255, 255), thickness=THICKNESS)
    cv2.imwrite(str(store_files / "sk_head" / f"{gen_name}_head_skel.png"), head)

    if PLT:
        plt.figure()
        plt.imshow(head)


# %%
