# %%
# %load_ext autoreload
# %autoreload 2

import os
import sys

import numpy as np

os.environ["MKL_THREADING_LAYER"] = "GNU"

# import numpy as np
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer  # , LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

# from PIL import Image
# import datetime



# from torch import nn
# from torch.nn import functional as F
# from torch.utils.data import DataLoader, random_split
# from torchmetrics import Accuracy, F1Score, ConfusionMatrix
# from torchvision import transforms

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
else:
    prefix = "../"  # or "../"

sys.path.append(f"{prefix}src")
# from utils import read_pretrained_model
# from HummingbirdLoader import HeronLoader, Denormalize
from MZBModel import MZBModel

seed = 555
np.random.seed(seed)  # apply this seed to img tranfsorms
torch.manual_seed(seed)  # needed for torchvision 0.7

# %%
if __name__ == "__main__":
    # scripts/Lit_hummingbird_finetune.py --batch_size=185 --data_dir=data/bal_cla_diff_loc_all_vid/
    # --learning_rate=0.00010856749693422446
    # --num_workers_loader=20 --pretrained_network=resnet18

    # Define checkpoints callbacks
    # best model on validation
    best_val_cb = pl.callbacks.ModelCheckpoint(
        filename="best-val-{epoch}-{step}-{val_loss:.1f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # latest model in training
    last_mod_cb = pl.callbacks.ModelCheckpoint(
        filename="last-{step}", every_n_train_steps=50, save_top_k=1
    )

    # Define progress bar callback
    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

    # %%
    model = MZBModel(
        data_dir=Path(
            f"{prefix}data/learning_sets/"
        ),  # bal_cla_diff_loc_all_vid/", "double_negs_bal_cla_diff_loc_all_vid/"
        pretrained_network="efficientnet_b2",  # "resnet18", # resnet50 # efficientnet-
        learning_rate=1e-2,  # 1e-6
        batch_size=64,  # 128
        weight_decay=0,  # 1e-3
        num_workers_loader=16,
        step_size_decay=25,
    )

    # %%
    name_run = "mzb-proto-effnetb1-v2"  # f"{model.pretrained_network}"
    cbacks = [pbar_cb, best_val_cb, last_mod_cb]
    wb_logger = WandbLogger(project="mzb-pil", name=name_run if name_run else None)
    wb_logger.watch(model, log="all")
    # TensorBoardLogger("tb_logs", name="")

    trainer = Trainer(
        gpus=-1,  # [0,1],
        max_epochs=250,
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=16,
        callbacks=cbacks,
        auto_lr_find=False,  #
        auto_scale_batch_size=False,
        logger=wb_logger,
        replace_sampler_ddp=False,
        log_every_n_steps=1
        # profiler="simple",
    )

    trainer.fit(model)

    # %%
    # %%
    # from matplotlib import pyplot as plt
    # from MZBLoader import Denormalize
    # import numpy as np

    # tr = model.train_dataloader()

    # denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # x, y, ind = tr.dataset.__getitem__(20)

    # print(x.shape)
    # print(y)
    # print(ind)

    # plt.figure()
    # plt.imshow(denorm(np.transpose(x,(1,2,0))))
