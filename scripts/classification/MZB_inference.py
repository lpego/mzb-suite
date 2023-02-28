# %% 
%load_ext autoreload
%autoreload 2

import os
import sys

os.environ['MKL_THREADING_LAYER'] = 'GNU'

from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from PIL import Image
from skimage import exposure
from sklearn.metrics import classification_report
from torchmetrics import ROC, ConfusionMatrix, F1Score, PrecisionRecallCurve

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
else:
    prefix = "../"  # or "../"
    
sys.path.append(f"{prefix}src")

from MZBLoader import Denormalize
from MZBModel import MZBModel
# from src.utils import read_pretrained_model, find_checkpoints
from utils import find_checkpoints, read_pretrained_model

# %% 
# dirs = find_checkpoints(Path(f"{prefix}mzb-pil"), version="rws33fgd", log="last")#.glob("**/*.ckpt")) 
# dirs = find_checkpoints(Path(f"{prefix}mzb-pil"), version="u8q4m6mh", log="last")#.glob("**/*.ckpt")) 
# dirs = find_checkpoints(Path(f"{prefix}mzb-pil"), version="z9gw5d0r", log="last")#.glob("**/*.ckpt")) 
# dirs = find_checkpoints(Path(f"{prefix}mzb-pil"), version="dws3lu2j", log="last")#.glob("**/*.ckpt")) 
dirs = find_checkpoints(Path(f"{prefix}mzb-pil"), version="a9zbuzsu", log="last")#.glob("**/*.ckpt")) 

mod_path = dirs[0]

model = MZBModel()  
model = model.load_from_checkpoint( 
        checkpoint_path=mod_path,
        # hparams_file= str(mod_path.parents[1] / 'hparams.yaml') #same params as args
    )

model.data_dir = Path(f"{prefix}data/learning_sets/")

model.eval()
# %%
# dataloader = model.train_dataloader(shuffle=False)
# dataloader = model.dubendorf_dataloader()
dataloader = model.val_dataloader()

pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

trainer = pl.Trainer(
    max_epochs=1,
    gpus=1, #[0,1],
    callbacks=[pbar_cb], 
    enable_checkpointing=False,
    logger=False
)

outs = trainer.predict(model=model, dataloaders=[dataloader], return_predictions=True)

# %% 
files = [folds for folds in sorted(list((model.data_dir / "val_set").glob("*"))) if "zz_" not in folds.name]

dir_dict_trn = {}
for a in files:
    dir_dict_trn[a.name] = a

for i, k in enumerate(dir_dict_trn):
    print(f"class {i}: {k} --  N = {len(list(dir_dict_trn[k].glob('*.png')))}")

# %% 
y = []; p = []; gt = []
for out in outs: 
    y.append(out[0].numpy().squeeze())
    p.append(out[1].numpy().squeeze())   
    gt.append(out[2].numpy().squeeze())

try:
    yc = np.concatenate(y)
    pc = np.concatenate(p)
    gc = np.concatenate(gt)

except: 
    yc = np.array(y)
    pc = np.asarray(p)
    gc = np.asarray(gt)


mzb_class = 6

denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

unc_score = -pc[:,mzb_class]
# unc_score = np.sum(pc * np.log(pc), axis=1)
# unc_score = np.sort(pc,axis=1)[:,-1] - np.sort(pc,axis=1)[:,-2]

# ss = np.argsort(-pc[:, mzb_class])
ss = np.argsort(unc_score)
sub = ss 
# sub = ss[gc[ss] == mzb_class]

files = dataloader.dataset.img_paths

plt.figure(figsize=(15,5))
for c in np.unique(gc):
    # plt.hlines(gc[gc==c], xmin=np.where(gc==c)[0][0], xmax=np.where(gc==c)[0][-1])
    plt.scatter(np.where(gc==c)[0], np.ones_like(gc[gc==c]), label=f"{c}: {list(dir_dict_trn.keys())[c]}")
plt.plot(pc)
plt.legend()

# %% 
print(f"PREDICTING CLASS {list(dir_dict_trn.keys())[mzb_class]}") 

DETECT = 0
DIFF = True
FR = 0; N = 10

preds = []

for i, ti in enumerate(sub):
    if i < FR:
        continue

    fi = files[ti]
    im = Image.open(fi).convert("RGB")
    x = model.transform_ts(im)
    x = x[np.newaxis,...]

    with torch.set_grad_enabled(False):
        p = torch.softmax(model(x), dim = 1).cpu().numpy()
        pl_im = denorm(np.transpose(x.squeeze(),(1,2,0)))

    f, a = plt.subplots(1,1,figsize=(4,4))
    p_class = np.argmax(pc[ti,:])

    a.imshow(pl_im)
    a.axis("off")
    a.set_title(f"Inference: predicted class {p_class} with P {pc[ti, p_class]:.2f}\n"\
    f"{files[ti]} \n GT {gc[ti]},  "\
    f"Y @ {pc[ti,mzb_class]:.2f}")
    
    # if pc[ti, mzb_class] < 0.01: 
        # break

    if i > N:
        break
# %% 
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             plot_confusion_matrix)

names = list(dir_dict_trn.keys())

cmat = confusion_matrix(gc, np.argmax(pc,axis=1), normalize="true")

plt.figure(figsize=(7,7))
plt.imshow(cmat)
plt.xticks(range(len(names)), names, rotation=90)
plt.yticks(range(len(names)), names, rotation=0)
plt.colorbar()

f = plt.figure(figsize=(10,10))
aa = f.gca() 
IC = type('IdentityClassifier', (), {"predict": lambda i : i, "_estimator_type": "classifier"})
plot_confusion_matrix(IC, yc, gc, values_format=".1f", ax=aa,
     normalize=None, xticks_rotation="vertical", display_labels=names)

# cmplot = ConfusionMatrixDisplay(cmat, display_labels=list(dir_dict_trn.keys()))
# plt.figure(figsize=(10,10))
# cmplot.plot()
# %% 
# print(classification_report(pc[:,1] > 0.5, gc))
# print(classification_report(sortsco > 0.5, gc))
# print(ConfusionMatrix(num_classes=2)(torch.tensor(pc[:,1]) > 0.5 , torch.tensor(gc)).numpy())
# print(ConfusionMatrix(num_classes=2)(torch.tensor(sortsco) > 0.5, torch.tensor(gc)).numpy())
# %% 
# TO CHECK:
# - data/learning_sets/ephemeroptera/31_b1_ephemeroptera_01_clip_19_rgb.png 
# - data/learning_sets/brachyptera/32_bd_brachyptera_01_clip_2_rgb.png is equal to 
# data/learning_sets/plecoptera/32_bd_plecoptera_01_clip_3_rgb.png 
# - low scoring (trnset) of chironomidae
# - high scoring class 10 not in GT==10



# %%
# Alphabetical sort: \
# class 0: acari --  N = 3
# class 1: amphinemura --  N = 3
# class 2: baetidae --  N = 26
# class 3: baetis --  N = 10
# class 4: blephariceridae --  N = 1
# class 5: brachyptera --  N = 1
# class 6: chironomidae --  N = 56
# class 7: coleoptera --  N = 4
# class 8: diptera --  N = 31
# class 9: ephemerellidae --  N = 1
# class 10: ephemeroptera --  N = 25
# class 11: errors --  N = 133
# class 12: heptagenidae --  N = 57
# class 13: hydropsychidae --  N = 2
# class 14: isoperla --  N = 11
# class 15: leptophlebiidae --  N = 1
# class 16: leuctra --  N = 1
# class 17: leuctridae --  N = 13
# class 18: oligochaeta --  N = 2
# class 19: plecoptera --  N = 38
# class 20: protonemura --  N = 12
# class 21: simuliidae --  N = 56