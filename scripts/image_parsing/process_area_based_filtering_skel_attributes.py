# %%
import os
import pathlib
from pathlib import Path

import astropy.units as u
import cv2
import numpy as np
import pandas as pd
import skimage
from fil_finder import FilFinder2D
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
    PLOTS = True
else:
    prefix = Path("../")  # or "../"
    PLOTS = True

# from src import skeleton_tracer

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
area_class = {
    0: {"area": [0, 50000], "disk": 3, "lmode": "skeleton"},
    1: {"area": [50000, 150000], "disk": 5, "lmode": "skeleton"},
    2: {"area": [150000, 200000], "disk": 5, "lmode": "ellipse"},
    3: {"area": [200000, np.Inf], "disk": 9, "lmode": "ellipse"},
}

main_root = Path(
    prefix
    / "data/data_raw_custom_processing/project_portable_flume/clips_5k"  # , Mixed samples/Sample_site_1"
)

# import random

mask_proc = list(main_root.glob("**/*_mask.png"))
mask_proc.sort()

conv_rate = np.mean([133.1, 136.6, 133.2, 133.2, 133.2, 118.6, 133.4, 132.0])  # px / mm

SAVE_SKELETONS = main_root / "skeletons"
SAVE_SKELETONS.mkdir(exist_ok=True)

mask_props = pd.DataFrame(
    [],
    columns=[
        "clip_filename",
        "mean_rgb",
        "conv_rate_mm_px",
        "area",
        "skel_length",
        "skel_diam",
        "solidity",
        "area_over_perim" "ellipse_major",
        "ellipse_minor",
    ],
)

# %%
# os.system(f"touch {prefix}/test.test")
# a = 891
# for fi, fo in enumerate(mask_proc[a : a + 10]):
for fi, fo in enumerate(mask_proc):

    # fo = mprop.sample(n=1).clip_filename.iloc[0]
    # # ins = sub.sample(n=1)
    # fo = ins.clip_filename.iloc[0]
    # %
    mask_ = (cv2.imread(str(fo))[:, :, 0] / 255).astype(np.uint8)
    rgb_ = cv2.imread(str(fo)[:-8] + "rgb.png")[:, :, [2, 1, 0]].astype(np.uint8)
    # %

    pr_m = measure.regionprops(mask_)
    print(
        f"{fi+1}/{len(mask_proc)}, area {pr_m[0].area}: {fo.name}",
        end="\n",  # "\n"
    )

    # switch cases
    for aa in area_class:
        if area_class[aa]["area"][0] < pr_m[0].area < area_class[aa]["area"][1]:
            dpar = area_class[aa]["disk"]
            length_mode = area_class[aa]["lmode"]

    mask = binary_erosion(mask_, disk(dpar))  # & masf_
    mask = binary_dilation(mask, disk(dpar // 2))  # & masf_
    # mask = binary_erosion(mask, disk(dpar//2))  # & masf_
    # mask = binary_erosion(mask, disk(3))  # & masf_

    mask = mask.astype(np.uint8)
    subl = measure.label(mask)
    subp = measure.regionprops(subl)

    va = [a.area for a in subp]

    try:
        mm_ = subl == 1 + np.argmax(va)
        sha = subp[np.argmax(va)]
    except ValueError:
        mm_ = mask_
        sha = subp

    # mm_ = subl == np.argmax(va) + 1

    mm_ = mm_[..., np.newaxis]
    masked_im = rgb_ * np.concatenate((mm_, mm_, mm_), axis=2)
    # masked_im = masked_im.reshape(-1, 3)
    mean_col_rgb = np.mean(
        masked_im.reshape(-1, 3)[masked_im.reshape(-1, 3)[:, 0] != 0, :], axis=(0)
    )
    mm_ = mm_[:, :, 0]

    # populate full csv
    y0, x0 = sha.centroid
    rrf, ccf = skimage.draw.ellipse(
        int(y0),
        int(x0),
        int(sha.major_axis_length * 0.5),
        int(sha.minor_axis_length * 0.5),
        shape=mm_.shape,
        rotation=sha.orientation,
    )
    masf_ = np.zeros(mm_.shape, dtype=np.uint8)
    masf_[rrf, ccf] = 1

    x1 = x0 + np.cos(sha.orientation) * 0.5 * sha.minor_axis_length
    x3 = x0 - np.cos(sha.orientation) * 0.5 * sha.minor_axis_length
    y1 = y0 - np.sin(sha.orientation) * 0.5 * sha.minor_axis_length
    y3 = y0 + np.sin(sha.orientation) * 0.5 * sha.minor_axis_length

    x2 = x0 - np.sin(sha.orientation) * 0.5 * sha.major_axis_length
    y2 = y0 - np.cos(sha.orientation) * 0.5 * sha.major_axis_length
    x4 = x0 + np.sin(sha.orientation) * 0.5 * sha.major_axis_length
    y4 = y0 + np.cos(sha.orientation) * 0.5 * sha.major_axis_length

    skel_long = np.zeros(mask_.shape, dtype=np.uint8)
    rrl, ccl = skimage.draw.line(int(y2), int(x2), int(y4), int(x4))
    skel_long[rrl, ccl] = 1
    skel_long = skel_long & mm_

    skel_width = np.zeros(mask_.shape, dtype=np.uint8)
    rrl, ccl = skimage.draw.line(int(y1), int(x1), int(y3), int(x3))
    skel_width[rrl, ccl] = 1
    skel_width = skel_width & mm_

    if length_mode == "skeleton":
        filin = mm_ & masf_
        fil = FilFinder2D(filin.astype(np.uint8), mask=filin)
        fil.preprocess_image(flatten_percent=0)
        fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
        fil.medskel(verbose=False)
        fil.analyze_skeletons(
            branch_thresh=10 * u.pix,
            skel_thresh=100 * u.pix,
            prune_criteria="length",
            max_prune_iter=100,
        )
        skel_long = fil.skeleton_longpath.astype(int)  # & masf_
        maj_length = int(np.sum(skel_long))

        # skel_width = np.zeros(mask_.shape, dtype=np.uint8)
        # # x1 = x0 - 0.5 * np.cos(sha.orientation) * sha.minor_axis_length
        # # y1 = y0 + 0.5 * np.sin(sha.orientation) * sha.minor_axis_length
        # rrl, ccl = skimage.draw.line(int(y1), int(x1), int(y3), int(x3))
        # skel_width[rrl, ccl] = 1
        skel_width = skel_width & mm_
        min_length = int(np.sum(skel_width))

    elif length_mode == "ellipse":

        maj_length = np.sum(skel_long)
        min_length = np.sum(skel_width)
        # min_ax = np.sqrt((x3 - y3) ** 2 + (x1 - y1) ** 2)
        # maj_ax = np.sqrt((x2 - y2) ** 2 + (x4 - y4) ** 2)

        # maj_length = maj_ax if min_ax < maj_ax else min_ax
        # min_length = min_ax if min_ax < maj_ax else maj_ax

        # skel_long = np.zeros(mask_.shape, dtype=np.uint8)
        # rrl, ccl = skimage.draw.line(int(y2), int(x2), int(y4), int(x4))
        # skel_long[rrl, ccl] = 1 & mm_

        # linef_ = np.zeros(mask_.shape, dtype=np.uint8)
        # rrl, ccl = skimage.draw.line(int(y1), int(x1), int(y3), int(x3))
        # linef_[rrl, ccl] = 1 & mm_

    sub_df = pd.DataFrame(
        data={
            "clip_filename": str(fo),
            "mean_rgb": " ".join([f"{s}" for s in mean_col_rgb]),
            "conv_rate_mm_px": [conv_rate],
            "area": [sha.area],
            "skel_length": [maj_length],
            "skel_diam": [min_length],
            "ellipse_major": [sha.major_axis_length],
            "ellipse_minor": [sha.minor_axis_length],
            "solidity": [sha.solidity],
            "compactness": [4 * np.pi * sha.area / sha.perimeter**2],
        }
    )

    mask_props = pd.concat((mask_props, sub_df))

    mask_skel = skeletonize(mask_, method="zhang")
    medi_skel, distance = medial_axis(mask_, return_distance=True)

    if PLOTS:
        # add image mask + skeletons
        size_disk = np.max((10, pr_m[0].area // 20000))
        long_sk = binary_dilation(skel_long, disk(size_disk)) != 0
        short_sk = binary_dilation(skel_width, disk(size_disk)) != 0

        skels = long_sk + 1 * short_sk - (long_sk & short_sk)
        sub_col = ["red", "green"]
        subc = ListedColormap(sub_col)

        plt.figure()
        plt.imshow(mask_, cmap=plt.cm.gray)
        plt.axis("off")

        f, a = plt.subplots(1, 1, figsize=(5, 5))
        # skels = dilation(skels, disk(5))
        skels = skels.astype(float)
        skels[skels == 0] = np.nan
        # linef_ = linef_.astype(float)
        # linef_[linef_==0] = np.nan
        # a.imshow(masked_im)
        a.imshow(rgb_)
        a.imshow(skels, cmap=subc)
        # a.plot((x3, x1), (y3, y1), "-m", linewidth=2.5)
        # a.plot((x2, x4), (y2, y4), "-b", linewidth=2.5)
        a.axis("off")
        a.text(10, 40, f"{length_mode}, area: {sha.area}")
        plt.savefig(Path(str(fo)[:-8] + "measures.png"))

        # f, a = plt.subplots(1, 1, figsize=(5, 5))
        # a.imshow(mm_)
        # a.imshow(skels, cmap=subc)
        # # a.imshow(linef_, cmap='pink')
        # a.axis("off")
        # plt.tight_layout()

        # f, a = plt.subplots(1, 4, figsize=(10, 10))
        # a[0].imshow(masf_)
        # # a[0].plot((x3, x1), (y3, y1), "-m", linewidth=2.5)
        # # a[0].plot((x2, x4), (y2, y4), "-b", linewidth=2.5)
        # a[0].imshow(1+dilation(linef_,disk(5)))
        # a[1].imshow(mm_)
        # a[2].imshow(mask_)
        # a[3].imshow((mm_ & masf_))
        # plt.show()

        # linef_ = dilation(linef_, disk(11))
        # linef_ = linef_.astype(float)
        # linef_[linef_ == 0] = np.nan
        # f, a = plt.subplots(1, 1, figsize=(5, 5))
        # # a.imshow((masf_ * mm_))
        # a.imshow(rgb_)
        # a.imshow(linef_, cmap=subc)
        # a.axis("off")
        # plt.tight_layout()
    # %%

mask_props = mask_props.reset_index().drop(columns="index")
print(f"saving in {prefix}/FLUME_MASK_ATTRIBUTES_AREA_BASED_PARS_V4.csv")
mask_props.to_csv(f"{prefix}/FLUME_MASK_ATTRIBUTES_AREA_BASED_PARS_V4.csv")
# %%
