# %%
# import os, sys, time, copy


from pathlib import Path
from PIL import Image

# import datetime

import torch
import pytorch_lightning as pl

import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchmetrics import F1Score
from torchmetrics.functional import precision_recall
from torchvision import transforms

import segmentation_models_pytorch as smp

from mzb_workflow.skeletons.mzb_skeletons_dataloader import MZBLoader_skels, Denormalize

# %%
class MZBModel_skels(pl.LightningModule):
    """
    pytorch lightning class def and model setup
    """

    def __init__(
        self,
        data_dir="data/skel_segm/",
        pretrained_network="efficientnet-b2",
        learning_rate=1e-4,
        batch_size=32,
        weight_decay=1e-8,
        num_workers_loader=4,
        step_size_decay=5,
        num_classes=2,
    ):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = Path(data_dir)
        self.learning_rate = learning_rate
        self.architecture = pretrained_network
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers_loader = num_workers_loader
        self.step_size_decay = step_size_decay
        self.data_dir_tst = ""
        self.num_classes = num_classes

        # some written in stone stuff.
        self.im_folder = self.data_dir / "images"
        self.bo_folder = self.data_dir / "sk_body"
        self.he_folder = self.data_dir / "sk_head"

        np.random.seed(12)
        N = len(list(self.im_folder.glob("*.png")))
        self.trn_inds = sorted(
            list(np.random.choice(np.arange(N), size=int(0.8 * N), replace=False))
        )
        self.val_inds = sorted(list(set(np.arange(N)).difference(set(self.trn_inds))))
        self.size_im = 224
        self.dims = (3, self.size_im, self.size_im)
        # channels, width, height = self.dims

        self.transform_tr = transforms.Compose(
            [
                transforms.RandomRotation(degrees=[0, 360]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.ColorJitter(
                    brightness=[0.8, 1.2], contrast=[0.8, 1.2]
                ),  # (brightness=[0.75, 1.25], contrast=[0.75, 1.25]), # was 0.8, 1.5
                transforms.Resize(
                    (self.size_im, self.size_im),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.transform_ts = transforms.Compose(
            [
                # transforms.CenterCrop((self.size_im, self.size_im)),
                transforms.Resize(
                    (self.size_im, self.size_im),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),  # AT LEAST 224
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # We can try a small DeepLabV3+, eg densnet121 backbone?

        self.model = smp.Unet(
            encoder_name=self.architecture,
            encoder_weights="imagenet",
            in_channels=3,
            classes=2,
            activation=None,
        )

        # self.model = smp.DeepLabV3Plus(
        #     encoder_name=self.architecture,
        #     encoder_weights="imagenet",
        #     in_channels=3,
        #     classes=2,
        #     activation=None,
        # )

        # self.accuracy = smb.blabla()
        # if self.num_classes == 2:
        #     self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        # else:
        # self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
        self.loss_fn = smp.losses.TverskyLoss(
            smp.losses.MULTILABEL_MODE, alpha=0.3, beta=0.7
        )

        self.save_hyperparameters()

    def forward(self, x):
        "forward pass return unnormalised logits, normalise when needed"
        return self.model(x)

    def training_step(self, batch, batch_idx):
        "training iteration per batch"
        x, y, _ = batch

        logits = self(x)  # [:, 1, ...]

        loss = self.loss_fn(logits, y)
        self.log("trn_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        "validation iteration per batch"
        x, y, _ = batch
        logits = self(x)  # [:, 1, ...]

        loss = self.loss_fn(logits, y)

        self.log(f"val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, print_log: str = "tst"):
        "test iteration per batch"
        # Reuse the validation_step for testing
        return self.validation_step(batch, batch_idx, print_log)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = None):
        x, y, _ = batch
        logits = self.model(x)
        # print(shape)
        probs = torch.softmax(logits, dim=1)
        # preds = torch.argmax(probs, dim=1)
        return probs, y

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # The ReduceLROnPlateau scheduler requires a monitor

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt,
                    mode="min",
                    factor=0.1,
                ),
                "monitor": "val_loss",
                "frequency": self.step_size_decay
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     opt, T_max=self.step_size_decay
        # )
        # return [opt], [sch]

    ######################
    # DATA RELATED HOOKS #
    ######################

    def train_dataloader(self, shuffle=True):

        trn_d = MZBLoader_skels(
            self.im_folder,
            self.bo_folder,
            self.he_folder,
            learning_set="trn",
            ls_inds=self.trn_inds,
            transforms=self.transform_tr,
        )

        # number of draws from the weighted random samples matches the 2 * (n_positive // batch_size)
        return DataLoader(
            trn_d,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=self.num_workers_loader,
        )

    def val_dataloader(self):
        "def of custom val dataloader"

        val_d = MZBLoader_skels(
            self.im_folder,
            self.bo_folder,
            self.he_folder,
            learning_set="val",
            ls_inds=self.val_inds,
            transforms=self.transform_ts,
        )

        # number of draws from the weighted random samples matches the 2 * (n_positive // batch_size)
        return DataLoader(
            val_d,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers_loader,
        )

    def train_ts_augm_dataloader(self):

        trn_d = MZBLoader_skels(
            self.im_folder,
            self.bo_folder,
            self.he_folder,
            learning_set="trn",
            ls_inds=self.trn_inds,
            transforms=self.transform_ts,
        )

        # number of draws from the weighted random samples matches the 2 * (n_positive // batch_size)
        return DataLoader(
            trn_d,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers_loader,
        )

    def tst_dataloader(self):
        "def of custom test dataloader"
        return None

    def dubendorf_dataloader(self, data_dir):
        "def of custom test dataloader"
        dub_folder = Path(data_dir)

        tst_dube = MZBLoader_skels(
            dub_folder,
            Path(""),
            Path(""),
            learning_set="dubendorf",
            ls_inds=[],
            transforms=self.transform_ts,
        )

        return DataLoader(
            tst_dube,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers_loader,
        )
