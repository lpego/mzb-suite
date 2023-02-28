# %%
import matplotlib as mpl
import pandas as pd
from matplotlib import colors as pltc
from matplotlib import pyplot as plt

mpl.rc("figure", dpi=300)

import pathlib
from pathlib import Path

import cv2
import imutils
import numpy as np
import skimage
import skimage.segmentation
import yaml
from skimage import measure

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
else:
    prefix = "../"  # or "../"
# %%

main_root = Path(
    f"{prefix}data/data_raw_custom_processing/project_portable_flume/mixed_samples/sample_site_1/"
)

# main_root = Path("../data/2021_swiss_invertebrates/phenopype/dd_ponds/data/")
# main_root = Path(
#     f"{prefix}data/data_raw_custom_processing/dubendorf_ponds/Test_challenging_organisms/"
# )
files_proc = list(main_root.glob("**/*.csv"))
files_proc.sort()

# %%
summary = []
for i, fi in enumerate(files_proc[:]):
    print(f"{i+1}/{len(files_proc)}: {fi}")
    atts_ = pd.read_csv(fi).drop(columns=["Unnamed: 0"])
    summary.append(atts_)

summary = pd.concat(summary).reset_index().drop(columns=["index"])
colors = summary.average_color.apply(
    lambda x: np.asarray(
        eval(
            " ".join(x.split()).replace("[ ", "[").replace(" ]", "]").replace(" ", ", ")
        )
    )
)
colors = np.concatenate(colors).reshape(-1, 3)
summary["rgb_average"] = np.mean(colors, axis=1)
summary = pd.concat((summary, pd.DataFrame(colors, columns=["r", "g", "b"])), axis=1)

colors = summary.average_hsv.apply(
    lambda x: np.asarray(
        eval(
            " ".join(x.split()).replace("[ ", "[").replace(" ]", "]").replace(" ", ", ")
        )
    )
)
colors = np.concatenate(colors).reshape(-1, 3)
summary = pd.concat((summary, pd.DataFrame(colors, columns=["h", "s", "v"])), axis=1)

summary["area_ratio"] = summary.bbox_area / summary.area_px

# %%
if 1:
    plt.figure(figsize=(9, 4))
    plt.title("aspect ratio")
    plt.plot(summary.squareness, color="k")
    plt.fill_between(range(summary.shape[0]), y1=0.9, y2=1.1)
    plt.xticks(range(summary.shape[0]), summary.species, rotation=90, fontsize=5)

    plt.figure(figsize=(9, 4))
    plt.title("RGB averages")
    for i in range(3):
        plt.plot(summary.loc[:, ["r", "g", "b"]].iloc[:, i], color=["r", "g", "b"][i])

    plt.figure(figsize=(9, 4))
    plt.title("HSV")
    for i in range(3):
        plt.plot(summary.loc[:, ["h", "s", "v"]].iloc[:, i], color=["r", "g", "b"][i])
    plt.xticks(range(summary.shape[0]), summary.species, rotation=90, fontsize=5)

    # plt.figure(figsize=(9, 4))
    # plt.title("avg color std")
    # plt.plot(summary.average_color_std, color="k")
    # plt.fill_between(range(summary.shape[0]), y1=0, y2=50)
    # plt.xticks(range(summary.shape[0]), summary.species, rotation=90, fontsize=5)

    plt.figure(figsize=(9, 4))
    plt.title("area px, area bb")
    plt.plot(summary.area_px, color="k")
    plt.plot(summary.bbox_area, color="r")
    plt.xticks(range(summary.shape[0]), summary.species, rotation=90, fontsize=5)

    plt.figure(figsize=(9, 4))
    plt.title("area bb / area px")
    plt.plot(summary.area_ratio, color="k")
    plt.fill_between(range(summary.shape[0]), y1=1.5, y2=10)
    plt.xticks(range(summary.shape[0]), summary.species, rotation=90, fontsize=5)
# %%  filter out

ind_s = pd.concat(
    (
        summary.area_px < 5000,
        # summary.h < 50,
        # summary.v < 100,
        # summary.average_color_std > 65,
        # (summary.area_ratio < 1.5) | (summary.area_ratio > 10),
    ),
    axis=1,
)
summary_sub = summary.drop(index=np.where(np.any(ind_s, axis=1))[0]).copy()
summary_sub = summary_sub.reset_index(drop=True)
# %%
if 0:
    plt.figure(figsize=(9, 4))
    plt.title("aspect ratio")
    plt.plot(summary_sub.squareness, color="k")
    plt.fill_between(range(summary_sub.shape[0]), y1=0.9, y2=1.1)
    plt.xticks(
        range(summary_sub.shape[0]), summary_sub.species, rotation=90, fontsize=5
    )

    plt.figure(figsize=(9, 4))
    plt.title("RGB averages")
    for i in range(3):
        plt.plot(summary.loc[:, ["r", "g", "b"]].iloc[:, i], color=["r", "g", "b"][i])

    plt.figure(figsize=(9, 4))
    plt.title("HSV")
    for i in range(3):
        plt.plot(summary.loc[:, ["h", "s", "v"]].iloc[:, i], color=["r", "g", "b"][i])
    plt.xticks(range(summary.shape[0]), summary.species, rotation=90, fontsize=5)

    plt.figure(figsize=(9, 4))
    plt.title("area px, area bb")
    plt.plot(summary_sub.area_px, color="k")
    plt.plot(summary_sub.bbox_area, color="r")
    plt.xticks(
        range(summary_sub.shape[0]), summary_sub.species, rotation=90, fontsize=5
    )

    plt.figure(figsize=(9, 4))
    plt.title("area bb / area px")
    plt.plot(summary_sub.area_ratio, color="k")
    plt.fill_between(range(summary_sub.shape[0]), y1=1.5, y2=10)
    plt.xticks(
        range(summary_sub.shape[0]), summary_sub.species, rotation=90, fontsize=5
    )
    plt.show()
# %%
which_ind = summary_sub.bbox_area > 10  # .5e6
# which_ind = summary.average_color_std < 20
# which_ind = [True if "Notonectidae" in a else False for a in summary.species]
files_to_show = (
    summary_sub[which_ind].input_file.apply(lambda x: prefix + x[:-4]).unique()
)

# file_base = (
#     prefix
#     + "data/data_raw_custom_processing/dubendorf_ponds/Benthos_1C_Contr_140619/1C_Ceratopogonidae_01"
# )

for file_base in files_to_show:

    if ("difficulty" in file_base) or ("difficutly" in file_base):
        continue

    atts_ = pd.read_csv(file_base + "_props.csv").drop(columns=["Unnamed: 0"])
    colors = atts_.average_color.apply(
        lambda x: eval(
            " ".join(x.split()).replace("[ ", "[").replace(" ]", "]").replace(" ", ", ")
        )
    )
    colors = np.concatenate(colors).reshape(-1, 3)
    atts_["rgb_average"] = np.mean(colors, axis=1)

    colors = atts_.average_hsv.apply(
        lambda x: np.asarray(
            eval(
                " ".join(x.split())
                .replace("[ ", "[")
                .replace(" ]", "]")
                .replace(" ", ", ")
            )
        )
    )
    colors = np.concatenate(colors).reshape(-1, 3)
    atts_["hsv_h"] = colors[:, 0]
    atts_["area_ratio"] = atts_.bbox_area / atts_.area_px

    mask = cv2.imread(file_base + "_mask.png")[:, :, 0]
    im = cv2.imread(file_base + ".jpg")[:, :, [2, 1, 0]]

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im_t = hsv[:, :, 0].copy()
    im_t = ((im_t - np.min(im_t)) / (np.max(im_t) - np.min(im_t)) * 255).astype(
        np.uint8
    )
    im_t[im_t > 150] = 150
    im_t = ((im_t - np.min(im_t)) / (np.max(im_t) - np.min(im_t)) * 255).astype(
        np.uint8
    )

    colma = 255 * np.ones([mask.shape[0], mask.shape[1], 3], dtype=np.uint8)
    edges = np.zeros(mask.shape)
    loc = []
    mm = 1

    for lab in np.unique(mask):

        if lab == 1:
            continue

        # print(lab, colors[lab - 2, :])

        # print(atts_.iloc[lab - 2].squareness, 0.80 < atts_.iloc[lab - 2].squareness < 1.2)
        # if 0.9 < atts_.iloc[lab - 2].squareness < 1.1:
        #     continue

        # print(atts_.iloc[lab - 2].average_color_std, atts_.iloc[lab - 2].average_color_std > 40)
        if atts_.iloc[lab - 2].area_px < 5000:
            mask[mask == lab] = 1
            continue

        # if atts_.iloc[lab - 2].hsv_h < 50:
        #     continue

        # if (atts_.iloc[lab - 2].area_ratio < 1.5) | (
        #     atts_.iloc[lab - 2].area_ratio > 10
        # ):
        #     continue

        # colma[mask == lab] = atts_.average_color_std.iloc[lab - 2]
        # colma[mask == lab, :] = colors[lab-2,:].astype(np.uint8) # atts_.rgb_range.iloc[lab - 2]
        x, y = np.mean(np.where(mask == lab)[1]), np.mean(np.where(mask == lab)[0])
        loc.append([x, y, mm, lab - 2, atts_.iloc[lab - 2].png_mask_id])
        mm += 1

    loc = np.asarray(loc)
    
    # palette = plt.get_cmap('viridis').copy()
    # palette = palette.set_under('white', 1.0)  # 1.0 represents not transparent
    # levels = np.arange(0, len(np.unique(mask)), 1)
    # levels[0] = 1e-5
    # norm = pltc.BoundaryNorm(levels, ncolors=palette.N)
    my_cmap = plt.cm.get_cmap('viridis').copy()
    my_cmap.set_under('white')
    # my_cmap.set_over("white")
    # imshow(np.arange(25).reshape(5, 5),
    #     interpolation='none',
    #     cmap=my_cmap,
    #     vmin=.001)


    f, a = plt.subplots(1, 3, figsize=(15, 6))
    a[0].imshow(im)
    a[0].axis("off")
    a[1].imshow(mask-1, cmap=my_cmap, vmin=0.001, vmax=loc.shape[0])# plt.cm.viridis)
    a[1].axis("off")
    masked_image = im * np.transpose(np.tile((mask > 1), (3, 1, 1)), (1, 2, 0))
    a[2].imshow(masked_image, cmap=plt.cm.viridis)
    a[2].axis("off")
    # a[3].imshow(hsv[:, :, 2])

    for i in range(loc.shape[0]):
        a[1].text(
            loc[i, 0],
            loc[i, 1],
            str(np.uint8(loc[i, 2])),
            color="black",
            fontweight="bold",
        )
    
    plt.savefig(f"../outputs_examples/{'_'.join(file_base.split('/')[-2:])}.pdf")
    # break
    # f, a = plt.subplots(1, 4, figsize=(23, 12))
    # a[0].imshow(im)
    # a[1].imshow(hsv[:, :, 0], cmap=plt.cm.viridis)
    # a[2].imshow(hsv[:, :, 1], cmap=plt.cm.viridis)
    # a[3].imshow(hsv[:, :, 2], cmap=plt.cm.viridis)

    plt.show()
    # a[3].imshow(edges, cmap=plt.cm.viridis)
# %%

plt.figure(figsize=(9, 4))
plt.title("aspect ratio")
plt.plot(atts_.squareness, color="k")
plt.fill_between(range(atts_.shape[0]), y1=0.9, y2=1.1)
plt.xticks(range(atts_.shape[0]), atts_.png_mask_id, rotation=90, fontsize=10)
plt.grid("on")

plt.figure(figsize=(9, 4))
plt.title("RGB averages")
plt.plot(atts_.rgb_average)
plt.xticks(range(atts_.shape[0]), atts_.png_mask_id, rotation=90, fontsize=10)
plt.grid("on")

plt.figure(figsize=(9, 4))
plt.title("HSV H")
plt.plot(atts_.hsv_h)
plt.xticks(range(atts_.shape[0]), atts_.png_mask_id, rotation=90, fontsize=10)
plt.grid("on")

plt.figure(figsize=(9, 4))
plt.title("area px, area bb")
plt.plot(atts_.area_px, color="k")
plt.plot(atts_.bbox_area, color="r")
plt.xticks(range(atts_.shape[0]), atts_.png_mask_id, rotation=90, fontsize=10)
plt.grid("on")

plt.figure(figsize=(9, 4))
plt.title("area bb / area px")
plt.plot(atts_.area_ratio, color="k")
plt.fill_between(range(atts_.shape[0]), y1=1.5, y2=10)
plt.xticks(range(atts_.shape[0]), atts_.png_mask_id, rotation=90, fontsize=10)
plt.grid("on")

# %%
attributes = [
    "squareness",
    "average_color_std",
    "ell_minor_axis",
    "ell_major_axis",
    "bbox_area",
    "area_px",
]
plt.imshow(summary[attributes] / summary[attributes].max(axis=0), aspect="auto")

# %%
plt.plot(atts_.bbox_area / atts_.area_px)

# %%

# %%
plt.scatter(colors[:, 0], colors[:, 1])
plt.scatter(colors[:, 0], colors[:, 2])
plt.scatter(colors[:, 1], colors[:, 2])

# %%
