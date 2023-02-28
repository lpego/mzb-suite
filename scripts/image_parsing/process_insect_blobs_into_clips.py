# %%
# Test comment

from pathlib import Path

# from fil_finder import FilFinder2D
import astropy.units as u
import cv2
# import yaml
import numpy as np
# import pathlib
import pandas as pd
# %%
# import pandas as pd
from matplotlib import pyplot as plt
from skimage.morphology import medial_axis, skeletonize
from skimage.util import invert

# from skimage import measure, segmentation, morphology, feature
# import skimage

# from scipy import ndimage



try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
    PLOTS = False
else:
    prefix = "../"  # or "../"
    PLOTS = True


# %%
# main_root = Path(
#     f"{prefix}data/data_raw_custom_processing/project_portable_flume/"  # ,Mixed samples/Sample_site_1"
# )

main_root = Path(
    f"{prefix}data/data_raw_custom_processing/dubendorf_ponds/"  # ,Mixed samples/Sample_site_1"
)

files_proc = list(main_root.glob("**/*.png"))
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

SAVE_CLIPS = main_root / "clips_5k"
# classif_repo = Path(f"{prefix}../mzb-classification/data/")
# SAVE_CLIPS = classif_repo / "clips_dubendorf"
SAVE_CLIPS.mkdir(parents=True, exist_ok=True)

PREVIEW = True
# %%
for fi, fo in enumerate(files_proc[:]):
    # area = []
    list_ff = pd.read_csv(fo)
    for i, m_ in list_ff.iterrows():
        print(f"\r", end="\r")
        print(
            f"{fi+1}/{len(files_proc)}: {fo} -- proc {(i+1)} / {list_ff.shape[0]}",
            end="\n",
        )

        base_rgb = cv2.imread(prefix + m_.input_file)[:, :, [2, 1, 0]]
        mask = cv2.imread(prefix + m_.input_file[:-4] + "_mask.png")[:, :, [2, 1, 0]]
        if "project_portable_flume" in str(main_root):
            mask[location_cutout[0] :, location_cutout[1] :] = 0

        box = eval(m_.large_bb)
        rgb_cutout = base_rgb[box[1] : box[1] + box[3], box[0] : box[0] + box[2], :]
        mask_cutout = (
            mask[box[1] : box[1] + box[3], box[0] : box[0] + box[2], 0]
            == m_.png_mask_id
        )
        # mask_skel = skeletonize(mask_cutout, method="zhang")
        # medi_skel, distance = medial_axis(mask_cutout, return_distance=True)

        # filter area and and save
        a_ = np.sum(mask_cutout)
        if a_ > 5000:
            mask_cutout = (mask_cutout * 255).astype(np.uint8)
            name_clip = (
                m_.input_file.split("/")[-1].split(".")[0]
                + "_clip_"
                + str(m_.png_mask_id)
            )
            cv2.imwrite(
                str(SAVE_CLIPS / (name_clip + "_mask.png")),
                mask_cutout,
                # [cv2.IMWRITE_JPEG_QUALITY, 100],
            )

            rgb_cutout = rgb_cutout.astype(np.uint8)[:, :, [2, 1, 0]]
            name_clip = (
                m_.input_file.split("/")[-1].split(".")[0]
                + "_clip_"
                + str(m_.png_mask_id)
            )
            cv2.imwrite(
                str(SAVE_CLIPS / (name_clip + "_rgb.png")),
                rgb_cutout,
                # [cv2.IMWRITE_JPEG_QUALITY, 100],
            )

            # f, a = plt.subplots(1, 2, figsize=(10, 4))
            # a[0].imshow(rgb_cutout)
            # a[1].imshow(mask_cutout)
            # plt.show()

        # plt.figure()
        # plt.hist(area,bins=25)

        # %%
        #  https://fil-finder.readthedocs.io/en/latest/Filament2D_tutorial.html#
        # fil = FilFinder2D(mask_cutout.astype(np.uint8), mask=mask_cutout)
        # fil.preprocess_image(flatten_percent=0)
        # fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
        # fil.medskel(verbose=False)
        # fil.analyze_skeletons(
        #     branch_thresh=10 * u.pix, skel_thresh=100 * u.pix, prune_criteria="length"
        # )

        # # orien = []
        # # for f in fil.filaments:
        # #     f.rht_analysis()
        # #     orien = orien.append(f.orientation)

        # if PREVIEW:
        #     f, a = plt.subplots(1, 6, figsize=(20, 4))
        #     a[0].imshow(rgb_cutout)
        #     a[1].imshow(mask_cutout)
        #     a[2].imshow(fil.skeleton)
        #     a[2].contour(fil.skeleton_longpath, c="r")
        #     a[3].imshow(mask_skel)
        #     a[4].imshow(distance)
        #     a[5].imshow(medi_skel)
        #     plt.show()

        # img = cv2.imread(str(full_path_raw_image_in))
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# %%

# fil = FilFinder2D(mask_cutout.astype(np.uint8), distance=150 * u.pc, mask=mask_cutout)
# # fil.preprocess_image(flatten_percent=100)
# fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
# fil.medskel(verbose=False)
# fil.analyze_skeletons(branch_thresh=50 * u.pix, skel_thresh=0 * u.pix, prune_criteria='length')

# orien = []
# for f in fil.filaments:
#     f.rht_analysis()
#     orien.append(f.orientation)

# Show the longest path
# plt.imshow(fil.skeleton, cmap='gray')
# plt.contour(fil.skeleton_longpath, colors='r')
# plt.axis('off')
# plt.show()
# %%
