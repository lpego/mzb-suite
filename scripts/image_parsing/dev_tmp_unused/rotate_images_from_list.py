# %%# 
"""
    Script to rotate specific images 
"""


import pathlib
from pathlib import Path

import cv2
import imutils
import numpy as np
## 
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

main_root = Path(f"{prefix}data/data_raw_custom_processing/project_portable_flume/")

files_proc = [
    main_root / "site_number_31/High_Flow_2/31_HF2_Acari_01.jpg",
    main_root / "site_number_31/Benthos_1/31_B1_Heptagenidae_01.jpg",
    main_root / "site_number_31/Benthos_1/31_B1_Plecoptera_01.jpg",
    main_root / "Site_Number_32/Benthos_2/32_B2_Baetis_01.jpg",
]

files_proc.sort()

# %%

for file_base in files_proc:

    im = cv2.imread(str(file_base))
    im_tr = np.transpose(im, (1, 0, 2))[::-1, :, :]
    f, a = plt.subplots(1, 2, figsize=(12, 6))
    a[0].imshow(im)
    a[1].imshow(im_tr)
    plt.show()

    # mask = mask.astype(np.uint8)
    cv2.imwrite(str(file_base), im_tr, [cv2.IMWRITE_JPEG_QUALITY, 100])
# %%
