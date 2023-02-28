# %%
import pathlib
from pathlib import Path

import cv2
import imutils
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from skimage import measure

# %%

# main_root = Path("../data/2021_swiss_invertebrates/phenopype/dd_ponds/data/")
main_root = Path("../data/2021_swiss_invertebrates/phenopype/flume/base_drift/data/")

exp_folders = list(main_root.glob(f"*"))
exp_folders.sort()

# %%
cont_props = []
mask_props = []

for fo in exp_folders[:1]:

	cont_file = list(fo.glob("*contours*.csv"))[0]
	c_df_ = pd.read_csv(cont_file)
	c_df_["folder"] = cont_file.parents[0]
	c_df_["cent_x"] =  c_df_.center.apply(lambda x: x.split("(")[1].split(",")[0]).astype(int)
	c_df_["cent_y"] =  c_df_.center.apply(lambda x: x.split(")")[-2].split(",")[-1]).astype(int)
	
	cont_props.append(c_df_)

	mask_file = list(fo.glob("*masks*.csv"))[0]
	m_df_ = pd.read_csv(mask_file)
	mask_props.append(m_df_)

	dict_y = yaml.safe_load(open(fo / "attributes.yaml"))
	
	raw_image_in = Path("/data/shared/swiss-invertebrates-data/data/") / '/'.join(str(Path(pathlib.PureWindowsPath(dict_y["image_original"]["filepath"]))).split("/")[3:])

cont_props = pd.concat(cont_props).reset_index().drop(columns=["index"])
mask_props = pd.concat(mask_props).reset_index().drop(columns=["index"])


# %%

img = cv2.imread(str(raw_image_in)) 
plt.figure(figsize=(10,6))
plt.imshow(img[:,:,[2,1,0]],aspect="auto") 
plt.scatter(cont_props.cent_x, cont_props.cent_y, marker="*")

im_t = img.copy()
im_t = cv2.cvtColor(im_t, cv2.COLOR_BGR2GRAY)
im_t = cv2.GaussianBlur(im_t, (51, 51), 0)
thresh = im_t < 215

plt.figure(figsize=(10,6))
plt.imshow(thresh) 
# rec = mask_props.coords


# %%
labels = measure.label(thresh, connectivity=2, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")
# loop over the unique components
c = 1

exp_bb = 200 # px

for label in np.unique(labels):
	# if this is the background label, ignore it
	if label == 0:
		continue
	# otherwise, construct the label mask and count the
	# number of pixels 
	labelMask = np.zeros(thresh.shape, dtype=np.uint8)
	area = np.sum(labels == label)

	if area > 10000:
		labelMask[labels == label] = 255
		c += 1 
		# if the number of pixels in the component is sufficiently
		# large, then add it to our mask of "large blobs"
		# if numPixels > 300:
		# 	mask = cv2.add(mask, labelMask)
		cnts = cv2.findContours(labelMask.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		peri = cv2.arcLength(cnts[0], True)
		approx = cv2.approxPolyDP(cnts[0], 0.04 * peri, True)

		(x, y, w, h) = cv2.boundingRect(approx)
		(x, y, w, h) = (x-exp_bb,y-exp_bb, w+2*exp_bb, h+2*exp_bb)
		ar = w / float(h)
		
		print(f"width / height = {ar}, approx. square")
		f,a = plt.subplots(figsize=(10,6))
		a.imshow(img[:,:,[2,1,0]], aspect="auto") 
		rect = plt.Rectangle((x,y), w, h, fc="none",ec="black", linewidth=2)
		a.add_patch(rect)
		plt.show()


# %%
