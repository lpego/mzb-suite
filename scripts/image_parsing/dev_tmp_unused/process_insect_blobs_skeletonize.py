# %%
# Test comment

import pathlib
import time
from pathlib import Path

import astropy.units as u
import cv2
import imutils
import numpy as np
import pandas as pd
import skimage
import yaml
from fil_finder import FilFinder2D
# %%
# import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
from skimage import feature, measure, morphology, segmentation
from skimage.morphology import (binary_dilation, binary_erosion, dilation,
                                disk, erosion, medial_axis, skeletonize)
from skimage.util import invert

try:
    __IPYTHON__
except:
    prefix = Path("")  # or "../"
    PLOTS = False
else:
    prefix = Path("../")  # or "../"
    PLOTS = True

# %%
def parse_bad_masks(fpath):
    fi = Path(fpath)
    fi = fi.read_text()
    fi = ["_".join(a.split("_")[:-1]) for a in fi.split("\n")]
    return fi[1:-1]


# %%
main_root = Path(
    prefix
    / "data/data_raw_custom_processing/project_portable_flume/clips_5k"  # , Mixed samples/Sample_site_1"
)

# import random

mask_proc = list(main_root.glob("**/*_mask.png"))
mask_proc.sort()

fi_excl = parse_bad_masks(
    prefix
    / "data/data_raw_custom_processing/project_portable_flume/BadMasks_BadMeasures.txt"
)

conv_rate = np.mean([133.1, 136.6, 133.2, 133.2, 133.2, 118.6, 133.4, 132.0])  # px / mm

# from skimage.segmentation import mark_boundaries, slic, felzenszwalb, watershed
# norm = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
# if "project_portable_flume" in str(main_root):
#     location_cutout = [2750, 4900]

SAVE_SKELETONS = main_root / "skeletons"
SAVE_SKELETONS.mkdir(exist_ok=True)
# %%
# fdir = str(prefix) + "/data/data_raw_custom_processing/project_portable_flume/clips_5k/"
# for ni, fi in enumerate(fi_excl):
#     im = cv2.imread(fdir + fi + "_rgb.png")[:, :, [2, 1, 0]].astype(np.uint8)
#     me = cv2.imread(fdir + fi + "_measures.png")[:, :, [2, 1, 0]].astype(np.uint8)

#     f, a = plt.subplots(1,2)
#     a[0].imshow(im)
#     a[1].imshow(me)

#     plt.show()
#     # time.sleep(1)
#     if ni > 10:
#         break

# %%
# random.shuffle(mask_proc)
# fil.skeleton_longpath # sum, longest path on skeleton
# (linef_ & mask) # sum, approximation of width based on minor axis direction and surface intersected
# props.major_axis_length # length of ellipse major axis
# props.minor_axis_length # length of ellipse minor axis
mask_props = pd.DataFrame(
    [],
    columns=[
        "clip_filename",
        "mean_rgb",
        "conv_rate_mm_px",
        "area",
        "skel_lenght",
        "skel_diam",
        "solidity",
        "area_over_perim" "ellipse_major",
        "ellipse_minor",
    ],
)

for fi, fo in enumerate(mask_proc[:]):
    print(
        f"{fi+1}/{len(mask_proc)}: {fo}",
        end="\n",
    )
    mask_ = (cv2.imread(str(fo))[:, :, 0] / 255).astype(np.uint8)
    rgb_ = cv2.imread(fo + "rgb.png")[:, :, [2, 1, 0]].astype(np.uint8)

    mm_ = erosion(mask_, selem=disk(5))
    mm_ = mm_[..., np.newaxis]
    masked_im = rgb_ * np.concatenate((mm_, mm_, mm_), axis=2)
    # masked_im = masked_im.reshape(-1, 3)
    mean_col_rgb = np.mean(
        masked_im.reshape(-1, 3)[masked_im.reshape(-1, 3)[:, 0] != 0, :], axis=(0)
    )

    # mass-enclosing ellipse clip / preprocessing
    props = measure.regionprops(mask_)[0]
    # populate full csv
    y0, x0 = props.centroid
    rrf, ccf = skimage.draw.ellipse(
        int(y0),
        int(x0),
        int(props.major_axis_length * 0.5),
        int(props.minor_axis_length * 0.5),
        shape=mask_.shape,
        rotation=props.orientation,
    )
    masf_ = np.zeros(mask_.shape, dtype=np.uint8)
    masf_[rrf, ccf] = 1

    mask = binary_erosion(mask_, disk(11))  # & masf_
    subl = measure.label(mask)
    subp = measure.regionprops(subl)
    try:
        mask = subl == 1 + np.argmax([a.area for a in subp])
    except ValueError:
        mask = mask_

    # mask_skel = skeletonize(mask, method="zhang")
    # medi_skel, distance = medial_axis(mask, return_distance=True)

    #  https://fil-finder.readthedocs.io/en/latest/Filament2D_tutorial.html#
    fil = FilFinder2D(mask.astype(np.uint8), mask=mask)
    fil.preprocess_image(flatten_percent=0)
    fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(
        branch_thresh=10 * u.pix, skel_thresh=100 * u.pix, prune_criteria="length"
    )
    skel_long = fil.skeleton_longpath.astype(int) & masf_

    # approx width of insect as line ON MASK orthogonal to ellipse major axis
    x1 = x0 + 0.5 * np.cos(props.orientation) * props.minor_axis_length
    y1 = y0 - 0.5 * np.sin(props.orientation) * props.minor_axis_length

    linef_ = np.zeros(mask_.shape, dtype=np.uint8)
    rrl, ccl = skimage.draw.line(int(y0), int(x0), int(y1), int(x1))
    linef_[rrl, ccl] = 1

    x1 = x0 - 0.5 * np.cos(props.orientation) * props.minor_axis_length
    y1 = y0 + 0.5 * np.sin(props.orientation) * props.minor_axis_length
    rrl, ccl = skimage.draw.line(int(y0), int(x0), int(y1), int(x1))
    linef_[rrl, ccl] = 1
    linef_ = linef_ & binary_erosion(mask_, disk(3))

    sub_df = (
        {}
    )  # , columns=["input_file", "squareness", "average_color_std", "is_palette", "is_background", "tight_bb", "large_bb"])
    sub_df["clip_filename"] = str(fo)
    sub_df["mean_rgb"] = " ".join([f"{s}" for s in mean_col_rgb])
    sub_df["conv_rate_mm_px"] = conv_rate
    sub_df["area"] = props.area
    sub_df["skel_lenght"] = int(np.sum(skel_long))
    sub_df["skel_diam"] = int(np.sum(linef_))
    sub_df["ellipse_major"] = props.major_axis_length
    sub_df["ellipse_minor"] = props.minor_axis_length
    sub_df = pd.DataFrame(data=sub_df, index=[0])
    mask_props = mask_props.append(sub_df)

    if 1:
        # add image mask + skeletons
        size_disk = np.max((5, props.area // 20000))
        long_sk = binary_dilation(skel_long, disk(size_disk)) != 0
        short_sk = binary_dilation(linef_, disk(size_disk)) != 0
        skels = 2 * long_sk + short_sk - (long_sk & short_sk)
        sub_col = ["red", "green"]
        subc = ListedColormap(sub_col)
        f, a = plt.subplots(1, 1, figsize=(5, 5))
        # skels = dilation(skels, disk(5))
        skels = skels.astype(float)
        skels[skels == 0] = np.nan
        # a.imshow(masked_im)
        a.imshow(rgb_)
        a.imshow(skels, cmap=subc)
        a.axis("off")
        plt.tight_layout()
        plt.savefig(Path(str(fo)[:-8] + "measures.png"))
        plt.close("all")


# %%
mask_props = mask_props.reset_index().drop(columns="index")
mask_props.to_csv("../FLUME_MASK_ATTRIBUTES.csv")
# %%
