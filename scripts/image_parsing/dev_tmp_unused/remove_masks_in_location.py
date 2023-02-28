# %%
import pathlib
from pathlib import Path

import cv2
import imutils
import numpy as np
import pandas as pd
import skimage
import skimage.segmentation
import yaml
from matplotlib import pyplot as plt
from skimage import measure

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
else:
    prefix = "../"  # or "../"
# %%

main_root = Path(
    f"{prefix}data/data_raw_custom_processing/project_portable_flume/"  # ,Mixed samples/Sample_site_1"
)
# main_root = Path(f"{prefix}data/data_raw_custom_processing/")
files_proc = list(main_root.glob("**/*_mask.png"))
files_proc.sort()

# %%
location_cutout = [2750, 4900]
for file_base in files_proc[:10]:
    print(str(file_base))

    mask = cv2.imread(str(file_base))
    supr = mask.copy()
    supr[location_cutout[0] :, location_cutout[1] :, :] = np.max(mask) + 1

    f, a = plt.subplots(1, 1, figsize=(12, 6))
    a.imshow(supr)
    plt.show()


# %%
