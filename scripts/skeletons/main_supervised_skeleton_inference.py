# %%
# Models available in this current version:
# FOR BODY
# dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="ra43x1rn", log="best")#.glob("**/*.ckpt"))
# FOR HEAAD
# dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="w1i7bxpf", log="best")#.glob("**/*.ckpt"))
# BOTH
# dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="rns3epkx", log="last") # effnetb2
# dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="t88exedf", log="last") # resnet 18
# dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="p4fjth2e", log="last") # resnet 34
# dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="7vowykvv", log="best") # resnet 50
# dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="z5jx9gtb", log="best") # resnet 34 deeplabV3+
# dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="y4f351yv", log="last") # mit_b2
# dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="hhse6alx", log="best") # mit_b4
# dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="33azu4vn", log="best") # mit_b2 tversky loss v1
# dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="pt72ofvi", log="best") # mit_b2 tversky loss v2
# dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="i6vl2f2j", log="last") # mit_b1 tversky loss

import argparse
import os
import sys
import torch
import cv2

from datetime import datetime
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from PIL import Image
from skimage.morphology import thin
from torchvision import transforms

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
else:
    prefix = "../../"  # or "../"

sys.path.append(f"{prefix}")

from mzb_workflow.skeletons.mzb_skeletons_pilmodel import MZBModel_skels
from mzb_workflow.skeletons.mzb_skeletons_helpers import paint_image_tensor, Denormalize
from mzb_workflow.utils import cfg_to_arguments, find_checkpoints

# Set the thread layer used by MKL
os.environ["MKL_THREADING_LAYER"] = "GNU"

# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file",
    type=str,
    required=True,
    help="path to config file",
)
parser.add_argument(
    "--input_dir",
    type=str,
    required=True,
    help="path with images for inference",
)
parser.add_argument(
    "--input_model",
    type=str,
    required=True,
    help="path to model checkpoint",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="where to save skeleton measure predictions as csv",
)
parser.add_argument("--verbose", "-v", action="store_true", help="print more info")
args = parser.parse_args()

# args = {}
# args["config_file"] = f"{prefix}configs/global_configuration.yaml"
# args[
#     "input_dir"
# ] = f"{prefix}data/learning_sets/project_portable_flume/skeletonization/"
# args["input_model"] = f"{prefix}models/mzb-skels/i6vl2f2j/"
# args[
#     "output_dir"
# ] = f"{prefix}results/skeletons/project_portable_flume/supervised_skeletons/"
# args["verbose"] = True
# args = cfg_to_arguments(args)

with open(str(args.config_file), "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg = cfg_to_arguments(cfg)

if cfg.skel_save_sup_masks is not None:
    print("ayo")
    cfg.skel_save_sup_masks = Path(f"{prefix}{cfg.skel_save_sup_masks}")
    cfg.skel_save_sup_masks.mkdir(parents=True, exist_ok=True)

args.output_dir = Path(args.output_dir)

if args.verbose:
    print(f"main args: {args}")
    print(f"scripts config: {cfg}")

# %%
dirs = find_checkpoints(
    Path(args.input_model).parents[0],
    version=Path(args.input_model).name,
    log=cfg.infe_model_ckpt,
)

mod_path = dirs[0]

model = MZBModel_skels()
model.model = model.load_from_checkpoint(
    checkpoint_path=mod_path,
)

model.data_dir = Path(args.input_dir)
model.im_folder = model.data_dir / "images"
model.bo_folder = model.data_dir / "sk_body"
model.he_folder = model.data_dir / "sk_head"

# this is unfortunately necessary to get the model to work, reindex trn/val split
np.random.seed(12)
N = len(list(model.im_folder.glob("*.png")))
model.trn_inds = sorted(
    list(np.random.choice(np.arange(N), size=int(0.8 * N), replace=False))
)
model.val_inds = sorted(list(set(np.arange(N)).difference(set(model.trn_inds))))
model.eval()
model.freeze()
# %%
# dataloader = model.train_dataloader(shuffle=False)
# dataloader = model.dubendorf_dataloader()
if "flume" in str(args.input_dir):
    dataloader = model.val_dataloader()
    dataset_name = "flume"
else:
    data_dir = Path(
        "/data/shared/mzb-classification/data/raw_learning_sets_duben/insects/"
    )
    dataloader = model.dubendorf_dataloader(data_dir)
    dataset_name = "dubendorf"

    # dataloader = model.external_dataloader(
    #     model.data_dir, glob_pattern=cfg.infe_image_glob
    # )

im_fi = dataloader.dataset.img_paths

pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

trainer = pl.Trainer(
    max_epochs=1,
    gpus=1,  # [0,1],
    callbacks=[pbar_cb],
    enable_checkpointing=False,
    logger=False,
)

outs = trainer.predict(model=model, dataloaders=[dataloader], return_predictions=True)
# %%
# aggregate predictions
p = []
gt = []
for out in outs:
    p.append(out[0].numpy())
    gt.append(out[1].numpy())
pc = np.concatenate(p)
gc = np.concatenate(gt)

# %%
cfg.skel_label_buffer_on_preds = 25
MASK = True if cfg.skel_label_buffer_on_preds else False
# nn body preds
preds_size = []

for i, ti in tqdm(enumerate(im_fi[:])):

    im = Image.open(ti).convert("RGB")

    # get original size of image for resizing predictions
    o_size = im.size

    # get predictions
    x = model.transform_ts(im)
    x = x[np.newaxis, ...]
    with torch.set_grad_enabled(False):
        p = torch.sigmoid(model(x)).cpu().numpy().squeeze()

    refined_skel = np.concatenate((p, np.zeros_like(p[0:1, ...])), axis=0)
    refined_skel = Image.fromarray(
        (255 * np.transpose(refined_skel, (1, 2, 0))).astype(np.uint8)
    )

    refined_skel = transforms.Resize(
        (o_size[1], o_size[0]),
        interpolation=transforms.InterpolationMode.BILINEAR,
    )(refined_skel)
    refined_skel = np.transpose(np.asarray(refined_skel), (2, 0, 1))

    # mask out the edges of the image
    if MASK:
        mask = np.ones_like(x[0, 0, ...])
        mask[-cfg.skel_label_buffer_on_preds :, :] = 0
        mask[: cfg.skel_label_buffer_on_preds, :] = 0
        mask[:, : cfg.skel_label_buffer_on_preds] = 0
        mask[:, -cfg.skel_label_buffer_on_preds :] = 0

        mask = Image.fromarray(mask)
        mask = np.array(
            transforms.Resize(
                (o_size[1], o_size[0]),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )(mask)
        )
        refined_skel = [
            (thin(a) > 0).astype(float) * mask for a in refined_skel[0:2, ...] > 50
        ]
    else:
        # Refine the predicted skeleton image
        refined_skel = [
            (thin(a) > 0).astype(float) for a in refined_skel[0:2, ...] > 50
        ]

    refined_skel = [(255 * s).astype(np.uint8) for s in refined_skel]

    if cfg.skel_save_sup_masks:
        name = "_".join(ti.name.split("_")[:-1])
        cv2.imwrite(
            str(cfg.skel_save_sup_masks / f"{name}_body.jpg"),
            refined_skel[0],
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        )
        cv2.imwrite(
            str(cfg.skel_save_sup_masks / f"{name}_head.jpg"),
            refined_skel[1],
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        )

    preds_size.append(
        pd.DataFrame(
            {
                "clip_name": "_".join(ti.name.split(".")[0].split("_")[:-1]),
                "nn_pred_body": [np.sum(refined_skel[0])],
                "nn_pred_head": [np.sum(refined_skel[1])],
            }
        )
    )

preds_size = pd.concat(preds_size)
out_dir = Path(
    f"{args.output_dir}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
)
out_dir.mkdir(exist_ok=True, parents=True)
preds_size.to_csv(out_dir / f"size_skel_supervised_model.csv", index=False)


if 0:
    # %%
    # %load_ext autoreload
    # %autoreload 2

    import os
    import sys

    os.environ["MKL_THREADING_LAYER"] = "GNU"

    from pathlib import Path

    import numpy as np
    import pandas as pd
    import pytorch_lightning as pl
    import torch
    from matplotlib import pyplot as plt
    from PIL import Image
    from skimage.morphology import skeletonize
    from skimage.util import invert
    from sklearn.metrics import classification_report
    from torchmetrics import ROC, ConfusionMatrix, F1Score, PrecisionRecallCurve
    from torchvision import transforms

    try:
        __IPYTHON__
    except:
        prefix = ""  # or "../"
    else:
        prefix = "../"  # or "../"

    sys.path.append(f"{prefix}src")

    from MZBLoader_skels import Denormalize
    from MZBModel_skels import MZBModel_skels

    # from src.utils import read_pretrained_model, find_checkpoints
    from utils import find_checkpoints, read_pretrained_model

    # %%

    # FOR BODY
    # dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="ra43x1rn", log="best")#.glob("**/*.ckpt"))
    # FOR HEAAD
    # dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="w1i7bxpf", log="best")#.glob("**/*.ckpt"))
    # BOTH
    # dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="rns3epkx", log="last") # effnetb2
    # dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="t88exedf", log="last") # resnet 18
    # dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="p4fjth2e", log="last") # resnet 34
    # dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="7vowykvv", log="best") # resnet 50
    # dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="z5jx9gtb", log="best") # resnet 34 deeplabV3+
    # dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="y4f351yv", log="last") # mit_b2
    # dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="hhse6alx", log="best") # mit_b4
    # dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="33azu4vn", log="best") # mit_b2 tversky loss v1
    # dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="pt72ofvi", log="best") # mit_b2 tversky loss v2
    # dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="i6vl2f2j", log="last") # mit_b1 tversky loss
    dirs = find_checkpoints(
        Path(f"{prefix}mzb-skels"), version="bprys4hk", log="last"
    )  # mit_b1 tversky loss longT
    # dirs = find_checkpoints(Path(f"{prefix}mzb-skels"), version="t8f84f0q", log="best") # effnetn2 tversky loss longT

    mod_path = dirs[0]
    print(mod_path)

    model = MZBModel_skels(
        data_dir=Path(
            f"{prefix}data/skel_segm/",
        ),
    )

    mm = model.load_from_checkpoint(
        checkpoint_path=mod_path,
        # hparams_file= str(mod_path.parents[1] / 'hparams.yaml') #same params as args
    )

    model.model = mm.model
    model.data_dir = Path(f"{prefix}data/skel_segm/")

    model.eval()
    # %%

    # dataloader = model.train_ts_augm_dataloader()
    # dataloader = model.val_dataloader()
    # dataset_name = "flume"

    data_dir = Path(
        "/data/shared/mzb-classification/data/raw_learning_sets_duben/insects/"
    )
    dataloader = model.dubendorf_dataloader(data_dir)
    dataset_name = "dubendorf"

    im_fi = dataloader.dataset.img_paths

    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

    trainer = pl.Trainer(
        max_epochs=1,
        gpus=1,  # [0,1],
        callbacks=[pbar_cb],
        enable_checkpointing=False,
        logger=False,
    )

    # %%
    outs = trainer.predict(
        model=model, dataloaders=[dataloader], return_predictions=True
    )

    # %%
    y = []
    p = []
    gt = []

    for out in outs:
        p.append(out[0].numpy())
        gt.append(out[1].numpy())

    pc = np.concatenate(p)
    gc = np.concatenate(gt)

    from typing import List, Tuple

    # ma_fi = dataloader.dataset.msk_paths\
    # %%
    import torch

    def paint_image(
        image: torch.Tensor, masks: torch.Tensor, color: List[float]
    ) -> torch.Tensor:
        """
        Given an input image, a binary mask indicating where to paint, and a color to use,
        returns a new image where the pixels within the mask are colored with the specified color.

        Args:
            image (torch.Tensor): Input image to paint.
            mask (torch.Tensor): Binary mask indicating where to paint.
            color (List[float]): List of 3 floats representing the RGB color to use.

        Returns:
            torch.Tensor: New image with painted pixels.
        """

        # Make a copy of the input image
        rgb_body = image.clone()

        c = 0
        for mask in masks:
            # Color the pixels within the mask with the specified color
            rgb_body[mask > 0.75] = torch.Tensor(
                [
                    color[c][0] * mask[mask > 0.75],
                    color[c][1] * mask[mask > 0.75],
                    color[c][2] * mask[mask > 0.75],
                ]
            ).permute((1, 0))
            c += 1
        # Return the new image
        return rgb_body

    # %%
    denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    DETECT = 0
    FR = 0
    N = 10
    BUFF = 20
    MASK = True
    sortind = np.random.randint(0, len(im_fi), N)
    # np.random.shuffle(im_fi)
    for i, ind in enumerate(sortind):

        ti = im_fi[ind]

        if i < FR:
            continue
        elif i > N:
            break

        # fi = files[ti]
        im = Image.open(ti).convert("RGB")
        x = model.transform_ts(im)
        x = x[np.newaxis, ...]

        if MASK:
            ma_ = "_".join(ti.name.split("_")[:-1]) + "_mask.png"
            mask = Image.open(Path(f"../data/clips_{dataset_name}/") / ma_).convert(
                "RGB"
            )
            mask = (denorm(model.transform_ts(mask)).numpy() > 0).astype(float)
        else:
            mask = np.ones_like(x[0, 0, ...])
            mask[-BUFF:, :] = 0
            mask[:BUFF, :] = 0
            mask[:, :BUFF] = 0
            mask[:, -BUFF:] = 0

        mask[mask == 0] = np.nan

        with torch.set_grad_enabled(False):
            p = torch.sigmoid(model(x)).cpu().numpy()

        pl_im = denorm(np.transpose(x.squeeze(), (1, 2, 0)))

        # Refine the predicted skeleton image
        refined_skel = np.concatenate(
            [(skeletonize(a) > 0).astype(float) * mask for a in p[0:1, ...] > 0.75]
        )
        # refined_skel[1,...] *= 2
        # refined_skel = np.sum(refined_skel, axis=0)*mask

        # prepare colored RGB images
        rgb_predictions = paint_image(pl_im, p[0, :, ...], [[1, 0, 0], [0, 1, 0]])
        # rgb_body = paint_image(pl_im, p[0,0,...], [0.8, 0.75, 0])
        # rgb_head = paint_image(pl_im, p[0,1,...], [0.8, 0.75, 0])
        rgb_skel = paint_image(pl_im, refined_skel, [[1, 0, 0], [0, 1, 0]])
        if dataloader.dataset.learning_set != "dubendorf":
            gplt = gc[ind, ...]
            rgb_gt = paint_image(pl_im, gplt, [[1, 0, 0], [0, 1, 0]])

        if dataloader.dataset.learning_set != "dubendorf":
            f, a = plt.subplots(1, 4, figsize=(20, 20))
            p_class = np.argmax(p)
            a[0].imshow(pl_im)
            # a[0].imshow(pl_im*np.concatenate(3*[mask[...,np.newaxis]], axis=2))
            a[0].axis("off")
            # a[1].imshow(p[0,0,...]*mask)
            a[1].imshow(rgb_predictions)
            a[1].axis("off")
            a[2].imshow(rgb_skel)
            # a[4].imshow(refined_skel)
            a[2].axis("off")
            # a[2].imshow(p[0,0,...]*mask)
            # a[2].imshow(rgb_head)
            # a[2].axis("off")
            # a[3].imshow(np.sum(gplt, axis=0))
            a[3].imshow(rgb_gt)
            a[3].axis("off")

        else:
            f, a = plt.subplots(1, 3, figsize=(20, 20))
            p_class = np.argmax(p)
            a[0].imshow(pl_im)
            # a[0].imshow(pl_im*np.concatenate(3*[mask[...,np.newaxis]], axis=2))
            a[0].axis("off")
            # a[1].imshow(p[0,0,...]*mask)
            a[1].imshow(rgb_predictions)
            a[1].axis("off")
            a[2].imshow(rgb_skel)
            # a[4].imshow(refined_skel)
            a[2].axis("off")
            # a[3].imshow(refined_skel)
            # a[3].axis('off')
    # we can concatenate the predictions for easy re-read
    # %%
    if 0:
        BUFF = 20
        MASK = True
        # nn body preds
        preds_size = []
        # np.random.shuffle(im_fi)
        for i, ti in enumerate(im_fi[:]):
            # fi = files[ti]
            im = Image.open(ti).convert("RGB")

            # ma_ = "_".join(ti.name.split("_")[:-1]) + "_mask.png"
            # mask = np.array(Image.open(Path("../data/clips_dubendorf/") / ma_).convert("RGB"))[:,:,0].astype(float) / 255

            o_size = im.size

            x = model.transform_ts(im)
            x = x[np.newaxis, ...]

            with torch.set_grad_enabled(False):
                p = torch.sigmoid(model(x)).cpu().numpy().squeeze()

            refined_skel = np.concatenate((p, np.zeros_like(p[0:1, ...])), axis=0)
            refined_skel = Image.fromarray(
                (255 * np.transpose(refined_skel, (1, 2, 0))).astype(np.uint8)
            )

            refined_skel = transforms.Resize(
                (o_size[1], o_size[0]),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )(refined_skel)
            refined_skel = np.transpose(np.asarray(refined_skel), (2, 0, 1))

            if MASK:
                # if SHAPE_MASK:
                # ma_ = "_".join(ti.name.split("_")[:-1]) + "_mask.png"
                # mask = Image.open(Path(f"../data/clips_{dataset_name}/") / ma_).convert("RGB")
                # mask = denorm(model.transform_ts(mask)).numpy()
                mask = np.ones_like(x[0, 0, ...])
                mask[-BUFF:, :] = 0
                mask[:BUFF, :] = 0
                mask[:, :BUFF] = 0
                mask[:, -BUFF:] = 0

                mask = Image.fromarray(mask)

                mask = np.array(
                    transforms.Resize(
                        (o_size[1], o_size[0]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    )(mask)
                )
                refined_skel = [
                    (skeletonize(a) > 0).astype(float) * mask
                    for a in refined_skel[0:2, ...] > 50
                ]
            else:
                # Refine the predicted skeleton image
                refined_skel = [
                    (skeletonize(a) > 0).astype(float)
                    for a in refined_skel[0:2, ...] > 50
                ]

            preds_size.append(
                pd.DataFrame(
                    {
                        "clip_name": "_".join(ti.name.split(".")[0].split("_")[:-1]),
                        "nn_pred_body": [np.sum(refined_skel[0])],
                        "nn_pred_head": [np.sum(refined_skel[1])],
                    }
                )
            )

        preds_size = pd.concat(preds_size)
        # annot = pd.read_csv("../data/merged_annotations_autom.csv")
        annot = pd.read_csv("../data/merged_annotations_autom.csv")

        annot = annot.set_index("clip_name")
        preds_size = preds_size.set_index("clip_name")
        preds_size = preds_size.merge(
            annot, left_index=True, right_index=True, how="inner"
        )

        err = pd.DataFrame()
        err["body_true_to_nn"] = preds_size["body_length"] - preds_size["nn_pred_body"]
        err["head_true_to_nn"] = preds_size["head_length"] - preds_size["nn_pred_head"]
        err["body_true_to_sk"] = preds_size["abs_error_bodysize_skel"]
        err["body_true_to_el"] = preds_size["abs_error_bodysize_ell"]
        err["head_true_to_el"] = preds_size["abs_error_headsize"]

        print(
            f"MSE NN to manual: \t body: {np.sqrt(np.mean(np.sum(err['body_true_to_nn']**2)))}"
        )
        print(f"\t \t  \t head: {np.sqrt(np.mean(np.sum(err['head_true_to_nn']**2)))}")

        print(
            f"MSE auto to manual: \t body skel: {np.sqrt(np.mean(np.sum(err['body_true_to_sk']**2)))}"
        )
        print(
            f"\t \t  \t body elli: {np.sqrt(np.mean(np.sum(err['body_true_to_el']**2)))}"
        )
        print(
            f"\t \t  \t head elli: {np.sqrt(np.mean(np.sum(err['head_true_to_el']**2)))}"
        )

    # %%
    # ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    # 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_32x16d',
    # 'resnext101_32x32d', 'resnext101_32x48d',
    # 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131',
    # 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
    # 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d',
    # 'se_resnext101_32x4d', 'densenet121',
    # 'densenet169', 'densenet201', 'densenet161', 'inceptionresnetv2', 'inceptionv4',
    # 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4',
    # 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'mobilenet_v2',
    # 'xception', 'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2',
    # 'timm-efficientnet-b3', 'timm-efficientnet-b4', 'timm-efficientnet-b5',
    # 'timm-efficientnet-b6', 'timm-efficientnet-b7', 'timm-efficientnet-b8',
    # 'timm-efficientnet-l2', 'timm-tf_efficientnet_lite0', 'timm-tf_efficientnet_lite1',
    # 'timm-tf_efficientnet_lite2', 'timm-tf_efficientnet_lite3',
    # 'timm-tf_efficientnet_lite4', 'timm-resnest14d', 'timm-resnest26d',
    # 'timm-resnest50d', 'timm-resnest101e', 'timm-resnest200e',
    # 'timm-resnest269e', 'timm-resnest50d_4s2x40d', 'timm-resnest50d_1s4x24d',
    # 'timm-res2net50_26w_4s', 'timm-res2net101_26w_4s', 'timm-res2net50_26w_6s',
    # 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s', 'timm-res2net50_14w_8s',
    # 'timm-res2next50', 'timm-regnetx_002', 'timm-regnetx_004', 'timm-regnetx_006',
    # 'timm-regnetx_008', 'timm-regnetx_016', 'timm-regnetx_032', 'timm-regnetx_040',
    # 'timm-regnetx_064', 'timm-regnetx_080', 'timm-regnetx_120', 'timm-regnetx_160',
    # 'timm-regnetx_320', 'timm-regnety_002', 'timm-regnety_004', 'timm-regnety_006',
    # 'timm-regnety_008', 'timm-regnety_016', 'timm-regnety_032', 'timm-regnety_040',
    # 'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120', 'timm-regnety_160',
    # 'timm-regnety_320', 'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d',
    # 'timm-mobilenetv3_large_075', 'timm-mobilenetv3_large_100', 'timm-mobilenetv3_large_minimal_100',
    # 'timm-mobilenetv3_small_075', 'timm-mobilenetv3_small_100', 'timm-mobilenetv3_small_minimal_100',
    # 'timm-gernet_s', 'timm-gernet_m', 'timm-gernet_l', 'mit_b0', 'mit_b1', 'mit_b2',
    # 'mit_b3', 'mit_b4', 'mit_b5', 'mobileone_s0', 'mobileone_s1', 'mobileone_s2',
    # 'mobileone_s3', 'mobileone_s4']"
