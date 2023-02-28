# %% Import libraries
### General utilities libraries
import pathlib
from pathlib import Path

### Computer Vision-related libraries
import cv2
import imutils
import numpy as np
import pandas as pd
import skimage
import yaml
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import feature, measure, morphology, segmentation

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

### Grab the list of images' filenames with the desired extension
files_proc = list(main_root.glob("**/*.jpg"))
files_proc.extend(list(main_root.glob("**/*.JPG")))
files_proc = [a for a in files_proc if "mask" not in str(a)]
files_proc.sort()

### Set normalization function and other project-specific parameters
# from skimage.segmentation import mark_boundaries, slic, felzenszwalb, watershed
norm = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

### Set the region of the images in the project where the colour scale is, so that it's easy to cut it out later
if "project_portable_flume" in str(main_root):
    location_cutout = [2750, 4900]

# files_proc = [
#     Path(
#         "../data/data_raw_custom_processing/project_portable_flume/Mixed samples/Sample_site_1/1_B1_mixed_02.JPG"
#     )
# ]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# %% 
print(f"1/{len(files_proc[:])}: {files_proc[0]}")

### Read in an image
img = cv2.imread(str(files_proc[0]))[:, :, [2, 1, 0]] # flip channels order: RGB -> BGR

mask_props = []

raw_image_in = files_proc[0]
full_path_raw_image_in = files_proc[0].resolve()

### 

newdir = str(full_path_raw_image_in.parent) + "/" + str(full_path_raw_image_in.stem)
os.makedirs(newdir, exist_ok=True)  # succeeds even if directory already exists.

### Convert to HSV colourspace, apply normalization
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
im_t = hsv[:, :, 0].copy() # keep only value channel of HSV
im_t = (255 * norm(np.mean(hsv[:, :, :2], axis=2))).astype(np.uint8)

### Use Gaussian blur to de-noise image, three passes
for _ in range(3):
    im_t = cv2.GaussianBlur(im_t, (21, 21), 0) # Gaussian convlution, kernel size 21x21, no std dev

# %% For the next step, we need to prepare a seed image first, starting form the original image
seed = np.copy(im_t)
seed[1:-1, 1:-1] = im_t.min() # assign minimum pixel value to all pixels but the outermost ones
mask = np.copy(im_t)

# %% 
### Dilation uses the seed image, which we set to the minimum intensity value of the original image, 
### and a mask image, in our case the blurred & normalized original image, which limits the maximum intensity value a pixel can take. 
### Dilation then spreads high pixel values (i.e. intensity) to neighbouring regions, up to the value permitted by the mask. 
### This effectively results in connecting neighbouring regions with high intensity, helping in connecting portions of the organisms together (e.g. appendages and body). 
dil = morphology.reconstruction(seed, im_t, method="dilation")
im_t = (im_t - dil).astype(np.uint8)

# %% This is where we apply the transformation that highlights foregroung VS background pixels
### It is composed of a local adaptive threshold (binary Gaussian) and a global threshold (Otsu): 
### the adaptive threshold is good for distinguishing edges when there is uneven brightness in different regions of the image; 
### the global threshold works best in finding the general shape of the organisms, but may lose some detail. 
ad_thresh = cv2.adaptiveThreshold(
        im_t, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 351, -2,
    ) # 255 is the maximum allowed pixel value, we choose a Gaussian average for computing the threshold value, and a binary threshold to indicate that we want either the min or max allowd values as output, we set a kernel size of 351, and set the constant (useful for trial-and-error fine tuning) to -2. 

t, thresh = cv2.threshold(im_t, 0, 255, cv2.THRESH_OTSU) # 0 and 255 are the min and max values allowed, respectively, and we specify a Otsu threshold, which automatically computes the global threshold value. 

# %% Combine together these thresholds, so that a pixel detected as foreground by either of the methods will be passed as foreground.  
thresh = thresh | ad_thresh # | is a bitwise OR

# %% Use simple morphological operations to fill gaps inside the segmented organisms. 
kernel = np.ones((11, 11), np.uint8) # create an "identity" kernel filled with "1", the size is the important parameter here 
for _ in range(5): # do five passes of the following
    thresh = cv2.morphologyEx(
        (255 * thresh).astype(np.uint8), cv2.MORPH_CLOSE, kernel # "close" (remove) outermost frame of masks, closing small blobs of pixels (i.e. unwanted noise), but also of organisms' silhouette
    )
    thresh = cv2.morphologyEx(
        (255 * thresh).astype(np.uint8), cv2.MORPH_OPEN, kernel # "open" (add) pixels to the ouotermost region of masks again, restoring original state of organisms' silhouettes (i.e. large blobs) but not noise (i.e. small blobs)
    )
    # thresh = cv2.dilate((255 * thresh).astype(np.uint8), kernel) # alternatively, dilate...
    # thresh = cv2.erode((255 * thresh).astype(np.uint8), kernel) # ...and erode have a similar effect

# %%
thresh = ndimage.binary_fill_holes(thresh) # fill holes between foreground pixels

# %% Since the scale is always in the same position, we can set the pixel value of that region to background
if "project_portable_flume" in str(main_root):
    thresh[location_cutout[0] :, location_cutout[1] :] = 0

# %% Connect regions that now touch each other and label them together
labels = measure.label(thresh, connectivity=2, background=0)

# %% Sanity check: plot the segmented images & original: 
if PLOTS:
    f, a = plt.subplots(1, 4, figsize=(21, 9))
    a[0].imshow(thresh) # combined threshold
    a[1].imshow(ad_thresh) # adaptive threshold
    a[2].imshow(img) # original image
    a[3].imshow(labels) # segmented & labelled image
    # plt.tight_layout()
    plt.show()
    # break

# %% 
rprop = measure.regionprops(labels) # save a bunch of measures from each mask
mask = np.ones(thresh.shape, dtype="uint8") # reset mask, overwriting it with a "blank" array of 1s

# %% Set parameters to suppress masks smallaer than exp_bb
c = 1 # initial pixel value for labelling individual masks
exp_bb = 200  # px, define bounding box expansion value

# %% 
sub_df = pd.DataFrame([]) # initialize empty pandas dataframe where to store measured features of the mask

for label in range(len(rprop)-120):  # for each mask in the image do:

    reg_pro = rprop[label] # grab the measures for that mask

    if reg_pro.label == 0: # if the mask is label "0" continue, otherwise exit loop
        continue
    if reg_pro.area < 200: # if the mask area is less than 200 (pixels) continue 
        continue
    # if reg_pro.bbox_area > 4e6:
    #     continue

    current_mask = np.zeros(thresh.shape) # create an array of all "background pixels" (0) of the size of the mask
    current_mask[labels == reg_pro.label] = 1  # transfer the "foreground pixels"

    # cnts = cv2.findContours(
    #     current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE # calculate countour hierarchy and save the outermost one, simple countour estimation
    # )
    # cnts = imutils.grab_contours(cnts) # grab the correct tuple based on opencv version

    # peri = cv2.arcLength(cnts[0], True) # define the perimeter of the mask
    # approx = cv2.approxPolyDP(cnts[0], 0.04 * peri, True) # approximate the shape with given epsilon parameter (peri)

    (min_row, min_col, max_row, max_col,) = reg_pro.bbox  # determine the bounding box for the mask; alternatively, cv2.boundingRect(approx) achives a similar result
    (x, y, w, h) = (min_col, min_row, max_col - min_col, max_row - min_row) # calculate the dimensions of the bounding box

    (x_e, y_e, w_e, h_e) = (
        np.max((x - exp_bb, 0)), # clamp to zero x-coordinate of size <200 px
        np.max((y - exp_bb, 0)), # clamp to zero y-coordinate of size <200 px
        w + 2 * exp_bb, # make it bigger
        h + 2 * exp_bb, # make it bigger
    )
    ar = w / float(h) # calculate aspect ratio

    ### Sanity check: plot bounding boxes
    if 1: # change this to "if 1:" to produce plots
        f, a = plt.subplots(1, 1, figsize=(10, 6)) 
        a.imshow(img[:, :, [0, 1, 2]], aspect="auto")
        rect = plt.Rectangle(
            (x_e, y_e), w_e, h_e, fc="none", ec="black", linewidth=2
        )
        a.add_patch(rect)
        plt.show()

    ### Define crop area for the various image versions: 
    crop = img[y_e : y_e + h_e, x_e : x_e + w_e, [2, 1, 0]] 
    crop_hsv = hsv[y_e : y_e + h_e, x_e : x_e + w_e, :]
    crop_mask = current_mask[y_e : y_e + h_e, x_e : x_e + w_e]
    crop_im_t = im_t[y_e : y_e + h_e, x_e : x_e + w_e]

    ### Now we take the portion of the original image contained in the bounding box and only keep pixels that are foreground in the mask: 
    im_crop_m = crop.reshape(-1, 3)[crop_mask.reshape(-1,).astype(bool), :] # in reshape(,), "-1" is a placeholder that automatically takes on the correct value to produce the desired output array dimentions... 
    hsv_crop_m = crop_hsv.reshape(-1, 3)[crop_mask.reshape(-1,).astype(bool), :] # ...then we subset this array based on the boolean value (True for foreground: any pixels >0; False for background: pixels =0)

    ### Calculate mean and standard deviation for each colour channel of the extracted mask
    im_crop_cmean = str(np.mean(im_crop_m, axis=0))
    hsv_crop_cmean = str(np.mean(hsv_crop_m, axis=0))

    im_crop_std = str(np.std(im_crop_m, axis=0))
    hsv_crop_std = str(np.std(hsv_crop_m, axis=0))

    # if reg_pro.bbox_area > (4e6): # if bounding box area is larger than 40k pixels, continue
    #     continue

    ### Assign an increasing value of "c" to the foreground pixels of each individual mask, acting as the label
    mask = mask + current_mask * c
    c += 1

    ### Sanity check: see if the masks assume different pixel values etc
    if PLOTS:
        f, a = plt.subplots(1, 4, figsize=(10, 6))
        a[0].imshow(crop)
        a[1].imshow(reg_pro.image)  # crop_mask)
        a[2].imshow(
            (crop * np.transpose(np.tile(crop_mask, (3, 1, 1)), (1, 2, 0))).astype(
                np.uint8
            )
        )
        im_t_crop_m = crop_im_t.reshape(-1, 1)[
            crop_mask.reshape(-1,).astype(bool), :
        ]
        a[3].hist(im_t_crop_m, bins=50)

        plt.show()

    ### Summarize mask info in the previously initialized CSV file: 
    sub_df = (
        {}
    )  # , columns=["input_file", "squareness", "average_color_std", "is_palette", "is_background", "tight_bb", "large_bb"])
    sub_df["input_file"] = raw_image_in # filepath
    sub_df["species"] = raw_image_in.name.split(".")[0] # assuming that the taxon name is in the filename, first position
    sub_df["png_mask_id"] = c # this is the pixel value assigned to the individual mask
    sub_df["reg_lab"] = reg_pro.label # original label of the mask, as per first segmentation
    sub_df["squareness"] = ar # aspect ratio
    sub_df["average_color_std"] = im_crop_std # in RGB space
    sub_df["average_color"] = im_crop_cmean # in RGB space
    sub_df["average_hsv"] = hsv_crop_cmean # in HSV space
    sub_df["average_hsv_std"] = hsv_crop_std # in HSV space
    sub_df["tight_bb"] = f"({x}, {y}, {w}, {h})" # tuple with coordinates of bounding box
    sub_df["large_bb"] = f"({x_e}, {y_e}, {w_e}, {h_e})" # tuple with coordinates of bounding box, expanded
    sub_df["ell_minor_axis"] = reg_pro.minor_axis_length # minor axis of ellipse encompassing the silhouette
    sub_df["ell_major_axis"] = reg_pro.major_axis_length # major axis of ellipse encompassing the silhouette
    sub_df["bbox_area"] = reg_pro.bbox_area # bounding box area
    sub_df["area_px"] = reg_pro.area # silhouetted organism's area
    sub_df["mask_centroid"] = str(reg_pro.centroid) # self-explanatory
    sub_df = pd.DataFrame(data=sub_df, index=[0]) # make pandas dataframe

    # print(sub_df)
    mask_props.append(sub_df) # append new rows to existing dataframe

    # print(np.max(mask), np.min(mask))
    # mask = (((mask - np.min(mask)) / (np.max(mask) - np.min(mask))) * 255).astype(
    #     np.uint8
    # )
    mask = mask.astype(np.uint8)

    # print(np.max(mask), np.min(mask))

    ### Sanity check: visualize the threshold and mask
    if PLOTS:
        f, a = plt.subplots(1, 3, figsize=(20, 16))
        a[0].imshow(thresh)
        a[1].imshow(img[:, :, [2, 1, 0]])
        a[2].imshow(mask)
        plt.show()
        break

    ### Write the whole segmented image to file, with individual masks labelled
    cv2.imwrite(
        str(full_path_raw_image_in)[:-4] + "_mask_all.png",
        mask,
        # [cv2.IMWRITE_JPEG_QUALITY, 100],
    )

    ### Write each individual mask as a separate image, 
    ### in a folder named after the original image (as not to clutter the parent directory)
    newdir = str(full_path_raw_image_in.parent) + "/" + str(full_path_raw_image_in.stem)
    os.makedirs(newdir, exist_ok=True)  # succeeds even if directory already exists.
    
    cv2.imwrite(
        newdir + "/" + str(full_path_raw_image_in.stem) + f"_mask_rgb_{c}.png", # generator for mask names 
        crop, # write individual mask
    )

    cv2.imwrite(
        newdir + "/" + str(full_path_raw_image_in.stem) + f"_mask_binary_{c}.png", # generator for mask names 
        crop, # write individual mask
    )

    ### Write to CSV the summary table of calculated features for each original image
    if mask_props:
        mask_props = pd.concat(mask_props).reset_index().drop(columns=["index"])
        mask_props.to_csv(str(full_path_raw_image_in.stem) + "_props.csv")

# %% 
