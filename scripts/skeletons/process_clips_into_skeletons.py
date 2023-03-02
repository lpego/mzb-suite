# %% test skimage skeletonize

# %load_ext autoreload
# %autoreload 2

import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk, medial_axis, thin
from tqdm import tqdm

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
    PLOTS = False
else:
    prefix = "../"  # or "../"
    PLOTS = False

sys.path.append(f"{prefix}src")

from skel_utils import (get_endpoints, get_intersections, paint_image,
                        segment_skel, traverse_graph)

# %%
SAVE_FILENAME = "FLUME_MASK_ATT_MYSKEL_v2.csv"
# load in file names that are classified as error by our CNN

err_filenames = sorted(
    list(
        Path(f"{prefix}../mzb-classification/data/raw_learning_sets_flume/errors").glob(
            "*.png"
        )
    )
)
err_filenames = ["_".join(a.name.split("_")[:-1]) + "_mask.png" for a in err_filenames]

# Load in all image clips we have for flume
main_root = Path(
    f"{prefix}data/data_raw_custom_processing/project_portable_flume/clips_5k"  # , Mixed samples/Sample_site_1"
)
mask_proc = sorted(list(main_root.glob("**/*_mask.png")))

# remove those that are errors
mask_proc = [a for a in mask_proc if a.name not in err_filenames]

# setup some area-specific parameters for filtering
area_class = {
    0: {"area": [0, 10000], "thinning": 1, "lmode": "skeleton"},
    2: {"area": [10000, 15000], "thinning": 11, "lmode": "skeleton"},
    3: {"area": [15000, 20000], "thinning": 13, "lmode": "skeleton"},
    4: {"area": [20000, 50000], "thinning": 13, "lmode": "skeleton"},
    5: {"area": [50000, 100000], "thinning": 15, "lmode": "skeleton"},
    6: {"area": [100000, np.inf], "thinning": 20, "lmode": "skeleton"},
}

conv_rate = np.mean([133.1, 136.6, 133.2, 133.2, 133.2, 118.6, 133.4, 132.0])  # px / mm

# %%
ddd = []
# Load the image
# PLOTS = True

for fo in tqdm(mask_proc[:]):
    # fo = mask_proc[131]
    # read in mask and rgb, rgb only for plotting

    mask_ = (cv2.imread(str(fo))[:, :, 0] / 255).astype(float)

    # Get needed filter size
    for aa in area_class:
        if area_class[aa]["area"][0] < np.sum(mask_) < area_class[aa]["area"][1]:
            dpar = area_class[aa]["thinning"]

    # Find the medial axis, threshold it and clean if multiple regions, keep largest
    maxis, distance = medial_axis(mask_, return_distance=True)
    mask_dist = distance > dpar
    regs = label(mask_dist)
    props = regionprops(regs)
    mask = regs == np.argmax([p.area for p in props if p.label > 0]) + 1

    # compute general skeleton by thinning the maks
    skeleton = thin(mask, max_num_iter=None)

    # get coordinates of point that intersect or are ends of the skeleton segments
    inter = get_intersections(skeleton=skeleton.astype(np.uint8))
    endpo = get_endpoints(skeleton=skeleton.astype(np.uint8))

    if PLOTS:
        rgb_ = cv2.imread(str(fo)[:-8] + "rgb.png")[:, :, [2, 1, 0]].astype(np.uint8)
        rgb_fi = paint_image(rgb_, skeleton, color=[255, 0, 0])
        rgb_ma = paint_image(rgb_, mask, color=[255, 0, 255])

    # case for which there are no segments (ie, only one)
    if len(inter) < 2:
        sub_df = pd.DataFrame(
            data={
                "clip_filename": fo.name,
                "conv_rate_mm_px": [conv_rate],
                "skel_length": [np.sum(skeleton)],
                "skel_length_mm": [np.sum(skeleton) / conv_rate],
                "segms": [[0]],
                "area": np.sum(mask_),
            }
        )
        ddd.append(sub_df)

        if PLOTS:
            f, a = plt.subplots(1, 2)
            a[0].imshow(rgb_fi)
            a[1].imshow(rgb_ma)
            plt.title(f"Area: {np.sum(mask_)}")

        # continue
    else:

        # filter out intersections that are closer than 1px and
        # segment the now intependently connected components

        # remove nodes that are too close and treat them as 1
        skel_labels, edge_attributes, skprop = segment_skel(skeleton, inter, conn=1)
        ds = distance_matrix(inter, inter) + 100 * np.eye(len(inter))
        duplicates = np.where(ds < 3)[0]
        try:
            inter = [a for a in inter if a != inter[duplicates[0]]]
        except:
            pass
        # skel_labels, edge_attributes, skprop = segment_skel(skeleton, inter, conn=2)

        intersection_nodes = []
        for coord in inter:
            local_cut = skel_labels[
                (coord[1] - 4) : (coord[1] + 5), (coord[0] - 4) : (coord[0] + 5)
            ]
            nodes_touch = np.unique(local_cut[local_cut != 0])
            intersection_nodes.append(list(nodes_touch))

        k = sorted(intersection_nodes)
        dedup = [k[i] for i in range(len(k)) if i == 0 or k[i] != k[i - 1]]
        intersection_nodes = dedup

        dead_ends = []
        for coord in endpo:
            dead_ends.append([skel_labels[coord[1], coord[0]]])
        dead_ends = sorted(dead_ends)

        graph = {}
        for nod in np.unique(skel_labels[skel_labels > 0]):
            nei = [a for a in intersection_nodes if nod in a]
            nei = [item for sublist in nei for item in sublist]
            graph[nod] = list(set(nei).difference([nod]))

        end_nodes = copy.deepcopy(dead_ends)
        end_nodes = [i for a in end_nodes for i in a]
        all_paths = []
        c = 0

        for init in end_nodes[:1]:
            p_i = traverse_graph(graph, init, end_nodes, debug=False)
            all_paths.extend(p_i)

        # clean remove doubles
        skel_cand = []
        for sk in all_paths:
            if sorted(sk) not in skel_cand:
                skel_cand.append(sorted(sk))

        # measure path lenghts and keep max one, that is the skel for you
        sk_l = []
        for sk in skel_cand:
            cus = 0
            for i in sk:
                cus += edge_attributes[i]
            sk_l.append(cus)

        sub_df = pd.DataFrame(
            data={
                "clip_filename": fo.name,
                "conv_rate_mm_px": [conv_rate],
                "skel_length": [sk_l[np.argmax(sk_l)]],
                "skel_length_mm": [sk_l[np.argmax(sk_l)] / conv_rate],
                "segms": [skel_cand[np.argmax(sk_l)]],
                "area": np.sum(mask_),
            }
        )
        ddd.append(sub_df)

        if PLOTS:
            f, a = plt.subplots(1, 3, figsize=(12, 12))
            a[0].imshow(
                paint_image(
                    skel_labels * 255, dilation(skel_labels > 0, disk(3)), [255, 0, 255]
                )
            )

            a[0].scatter(np.array(inter)[:, 0], np.array(inter)[:, 1])
            a[0].scatter(np.array(endpo)[:, 0], np.array(endpo)[:, 1], marker="s")
            for i in np.unique(skel_labels[skel_labels > 0]):
                a[0].text(
                    x=skprop[i - 1].centroid[1],
                    y=skprop[i - 1].centroid[0],
                    s=f"{i}",
                    color="white",
                )

            sel_skel = np.zeros_like(skel_labels)
            for i in np.unique(skel_labels[skel_labels > 0]):
                if i in skel_cand[np.argmax(sk_l)]:
                    sel_skel += dilation(skel_labels == i, disk(3))
            sel_skel = sel_skel > 0

            a[1].imshow(paint_image(rgb_fi, sel_skel, [255, 0, 0]))
            a[2].imshow(rgb_ma)

            a[0].title.set_text(f"Area: {np.sum(mask_)}")
            a[1].title.set_text(f"Sel Segm: {skel_cand[np.argmax(sk_l)]}")
            a[2].title.set_text(f"Skel_lenght_px {sk_l[np.argmax(sk_l)]}")


full_df = pd.concat(ddd)
print(full_df.shape)
full_df.to_csv(f"{prefix}{SAVE_FILENAME}.csv")

# %%
if 0:
    rgb_ = cv2.imread(str(fo)[:-8] + "rgb.png")[:, :, [2, 1, 0]].astype(np.uint8)
    rgb_fi = paint_image(rgb_, skeleton, color=[255, 0, 0])
    rgb_ma = paint_image(rgb_, mask, color=[255, 0, 255])

    labs = np.unique(skel_labels[skel_labels > 0])
    for i in labs:
        plt.figure()
        plt.imshow(
            paint_image(
                rgb_, dilation(skel_labels == i, disk(3)), [i / len(labs) * 255, 0, 255]
            )
        )
        plt.title(i)

    plt.figure()
    # plt.imshow(paint_image(rgb_, dilation(skel_labels, disk(3)), [255, 0, 255]))
    plt.imshow(dilation(skel_labels, disk(3)))
    plt.scatter(np.array(inter)[:, 0], np.array(inter)[:, 1])
    plt.scatter(np.array(endpo)[:, 0], np.array(endpo)[:, 1], marker="s")
    for i in np.unique(skel_labels[skel_labels > 0]):
        plt.text(
            x=skprop[i - 1].centroid[1],
            y=skprop[i - 1].centroid[0],
            s=f"{i}",
            color="white",
        )

    f, a = plt.subplots(1, 3, figsize=(12, 12))
    a[0].imshow(
        paint_image(
            skel_labels * 255, dilation(skel_labels > 0, disk(3)), [255, 0, 255]
        )
    )

    a[0].scatter(np.array(inter)[:, 0], np.array(inter)[:, 1])
    a[0].scatter(np.array(endpo)[:, 0], np.array(endpo)[:, 1], marker="s")
    for i in np.unique(skel_labels[skel_labels > 0]):
        a[0].text(
            x=skprop[i - 1].centroid[1],
            y=skprop[i - 1].centroid[0],
            s=f"{i}",
            color="white",
        )

    sel_skel = np.zeros_like(skel_labels)
    for i in np.unique(skel_labels[skel_labels > 0]):
        # if i in skel_cand[np.argmax(sk_l)]:
        sel_skel += dilation(skel_labels == i, disk(3))
    sel_skel = sel_skel > 0

    a[1].imshow(paint_image(rgb_fi, sel_skel, [255, 0, 0]))
    a[2].imshow(rgb_ma)

    a[0].title.set_text(f"Area: {np.sum(mask_)}")
    a[1].title.set_text(f"Sel Segm: {skel_cand[np.argmax(sk_l)]}")
    a[2].title.set_text(f"Skel_lenght_px {sk_l[np.argmax(sk_l)]}")
    # %%
