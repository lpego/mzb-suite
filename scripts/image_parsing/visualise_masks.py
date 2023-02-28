# %% 
# 

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# import imutils

# from skimage import measure, segmentation, morphology, feature

# import skimage
# from scipy import ndimage

# from fil_finder import FilFinder2D
# import astropy.units as u
# 
# from skimage.util import invert
# from skimage.morphology import medial_axis, skeletonize

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
    PLOTS = False
else:
    prefix = "../"  # or "../"
    PLOTS = True

# %%
main_root = Path(
    f"{prefix}data/data_raw_custom_processing/project_portable_flume/"  # ,Mixed samples/Sample_site_1"
)

# mixed_samples/sample_site_1

files_proc = list(main_root.glob("**/*.csv"))
files_proc = [a for a in files_proc if "difficulty" not in str(a)]
# files_proc.extend(list(main_root.glob("**/*.JPG")))
# files_proc = [a for a in files_proc if "mask" not in str(a)]
files_proc.sort()

# from skimage.segmentation import mark_boundaries, slic, felzenszwalb, watershed
norm = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
if "project_portable_flume" in str(main_root):
    location_cutout = [2750, 4750]

# files_proc = [
#     Path(
#         "../data/data_raw_custom_processing/project_portable_flume/Mixed samples/Sample_site_1/1_B1_mixed_02.JPG"
#     )
# ]
# %%
import matplotlib.colors as mcolors

for fi, fo in enumerate(files_proc[:1]):
    # area = []
    
    list_ff = pd.read_csv(fo)
    in_file = list_ff.iloc[0,:].input_file

    im = cv2.imread(prefix +in_file)[:, :, [2, 1, 0]]
    mask = cv2.imread(prefix + in_file[:-4] + "_mask.png")[:, :, [2, 1, 0]]
    loc = []
    mm = 1
    
    for lab in np.unique(mask):

        if lab == 1:
            continue

        if atts_.iloc[lab - 2].area_px < 5000:
            mask[mask == lab] = 0
            continue

        x, y = np.mean(np.where(mask == lab)[1]), np.mean(np.where(mask == lab)[0])
        loc.append([x, y, mm, lab - 2, atts_.iloc[lab - 2].png_mask_id])
        mm += 1

    loc = np.asarray(loc)


    cm = plt.cm.viridis(np.arange(len(np.unique(mask))))
    cm_len = len(cm)
    cm = plt.cm.viridis(range(500))
    cm[0,:] = [1,1,1,1]

    cm = ListedColormap(cm)
    
    f, a = plt.subplots(1, 3, figsize=(23, 12))
    a[0].imshow(im)
    a[0].axis("off")
    a[1].imshow(mask, cmap=cm, vmin=0.001, vmax=len(np.unique(mask)))# plt.cm.viridis)
    a[1].axis("off")
    masked_image = im * np.transpose(np.tile((mask > 1), (3, 1, 1)), (1, 2, 0))
    a[2].imshow(masked_image, cmap=plt.cm.viridis)
    a[2].axis("off")
    # a[3].imshow(hsv[:, :, 2])

    for i in range(loc.shape[0]):
        # a[0].text(
        #     loc[i, 0],
        #     loc[i, 1],
        #     str(np.uint8(loc[i, 2])),
        #     color="red",
        #     fontweight="bold",
        # )
        a[1].text(
            loc[i, 0],
            loc[i, 1],
            str(np.uint8(loc[i, 2])),
            color="white",
            fontweight="bold",
        )
        # a[2].text(
        #     loc[i, 0],
        #     loc[i, 1],
        #     str(np.uint8(loc[i, 2])),
        #     color="white",
        #     fontweight="bold",
        # )

    break
    # cm = plt.cm.viridis(range(500))
    # cm[0,:] = [1,1,1,1]
    
    # cm = ListedColormap(cm)
    # # cm = mcolors.LinearSegmentedColormap.from_list('colormap', cm)

    # plt.figure(figsize=(15,15))
    # plt.imshow(mask[:,:,0], vmin=0, vmax=cm_len)
    # plt.axis("off")
    # plt.colorbar()

    # plt.figure(figsize=(15,15))
    # plt.imshow(mask[1200:1300,1200:1300,0], cmap=cm)
    # plt.axis("off")

# %%
