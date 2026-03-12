import argparse
import os
import sys
import torch
import cv2

from datetime import datetime
from pathlib import Path
import pathlib
from PIL import Image
from matplotlib import pyplot as plt
from PIL import Image
from skimage.morphology import thin
from torchvision import transforms
from tqdm import tqdm

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml

from mzbsuite.skeletonization.mzb_skeletons_pilmodel import MZBModel_skels
from mzbsuite.skeletonization.mzb_skeletons_helpers import paint_image_tensor, Denormalize
from mzbsuite.utils import cfg_to_arguments, find_checkpoints

# Set the thread layer used by MKL
os.environ["MKL_THREADING_LAYER"] = "GNU"


def main(args, cfg):
    """
    Function to run inference of skeletons (body, head) on macrozoobenthos images clips, using a trained model.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace containing the arguments passed to the script. Notably:

            - input_dir: path to the directory containing the images to be classified
            - input_type: type of input data, either "val" or "external"
            - input_model: path to the directory containing the model to be used for inference
            - output_dir: path to the directory where the results will be saved
            - save_masks: path to the directory where the masks will be saved
            - config_file: path to the config file with train / inference parameters

    cfg : dict
        Dictionary containing the configuration parameters.

    Returns
    -------
    None. Saves the results in the specified folder.
    """

    torch.hub.set_dir("././models/hub/")

    dirs = find_checkpoints(
        Path(args.input_model).parents[0],
        version=Path(args.input_model).name,
        log=cfg.infe_model_ckpt,
    )

    mod_path = dirs[0]
    
    ### resolving Path in Windows
    if (sys.platform == "win32"):
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MZBModel_skels.load_from_checkpoint(
        checkpoint_path=mod_path,
        map_location=device,
        weights_only=False, # due to legacy checkpoint
    )
    
    model.to(device)

    model.data_dir = Path(args.input_dir)
    model.im_folder = model.data_dir / "images"
    model.bo_folder = model.data_dir / "sk_body"
    model.he_folder = model.data_dir / "sk_head"
    model.num_workers_loader = 4
    model.batch_size = 8

    # this is unfortunately necessary to get the model to work, reindex trn/val split
    np.random.seed(12)
    N = len(list(model.im_folder.glob("*.jpg")))
    model.trn_inds = sorted(
        list(np.random.choice(np.arange(N), size=int(0.8 * N), replace=False))
    )
    model.val_inds = sorted(list(set(np.arange(N)).difference(set(model.trn_inds))))
    model.eval()
    model.freeze()

    if args.input_type == "val":  # ("flume" in str(args.input_dir)) and
        dataloader = model.val_dataloader()
        dataset_name = "flume"
    elif args.input_type == "external":
        dataloader = model.external_dataloader(args.input_dir)
        dataset_name = "external"

    im_fi = dataloader.dataset.img_paths
    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=1)
    # Filter out mask files
    mask_suffixes = ("_msk.png", "_mask.jpeg", "_mask.jpg")
    im_fi = [p for p in im_fi if not any(str(p).endswith(suf) for suf in mask_suffixes)]

    trainer = pl.Trainer(
        precision=32,
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else 1,
        callbacks=[pbar_cb],
        enable_checkpointing=False,
        logger=False,
    )

    outs = trainer.predict(
        model=model, dataloaders=[dataloader], return_predictions=True
    )

    # aggregate predictions
    p = []
    gt = []
    for out in outs:
        p.append(out[0].numpy())
        gt.append(out[1].numpy())
    pc = np.concatenate(p)
    gc = np.concatenate(gt)

    # %%
    # nn body preds
    preds_size = []

    if args.verbose:
        print("Neural network predictions done, refining and saving skeletons...")

    for i, ti in tqdm(enumerate(im_fi), total=len(im_fi)):
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
        if (cfg.skel_label_buffer_on_preds > 0) and (not cfg.skel_label_clip_with_mask):
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
        elif cfg.skel_label_clip_with_mask:
            # load mask
            mask_insect = Image.open(
                cfg.glob_blobs_folder / ti.name[:-4] + "_mask.jpg"
            ).convert("RGB")
            mask_insect = np.array(mask_insect)[:, :, 0] > 0
            mask_insect = Image.fromarray(mask_insect)
            mask_insect = np.array(
                transforms.Resize(
                    (o_size[1], o_size[0]),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )(mask_insect)
            )
            refined_skel = [
                (thin(a) > 0).astype(float) * mask_insect
                for a in refined_skel[0:2, ...] > 50
            ]

        else:
            # Refine the predicted skeleton image
            refined_skel = [
                (thin(a) > 0).astype(float) for a in refined_skel[0:2, ...] > 50
            ]

        refined_skel = [(255 * s).astype(np.uint8) for s in refined_skel]

        if args.save_masks:
            name = "_".join(ti.name.split("_")[:-1])
            cv2.imwrite(
                str(args.save_masks / f"{name}_body.jpg"),
                refined_skel[0],
                [cv2.IMWRITE_JPEG_QUALITY, 100],
            )
            cv2.imwrite(
                str(args.save_masks / f"{name}_head.jpg"),
                refined_skel[1],
                [cv2.IMWRITE_JPEG_QUALITY, 100],
            )

        preds_size.append(
            pd.DataFrame(
                {
                    "clip_name": "_".join(ti.name.split(".")[0].split("_")[:-1]),
                    "nn_pred_body": [np.sum(refined_skel[0] > 0)],
                    "nn_pred_head": [np.sum(refined_skel[1] > 0)],
                }
            )
        )

    preds_size = pd.concat(preds_size)
    # out_dir = Path(
    #     f"{args.output_dir}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    # )
    out_dir = Path(f"{args.output_dir}")
    out_dir = (
        args.output_dir
        / f"{args.input_dir.name}_supervised_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    out_dir.mkdir(exist_ok=True, parents=True)

    preds_size.to_csv(out_dir / f"supervised_skeletons.csv", index=False)
    
    if (sys.platform == "win32"):
        pathlib.PosixPath = temp ### restore original pathlib function

if __name__ == "__main__":
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
        "--input_type",
        type=str,
        required=True,
        help="either 'val' or 'external'",
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
    parser.add_argument(
        "--save_masks",
        type=str,
        required=True,
        help="where to save skeleton masks predictions as jpg",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="print more info")
    args = parser.parse_args()

    with open(str(args.config_file), "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = cfg_to_arguments(cfg)

    if args.save_masks is not None:
        args.save_masks = Path(f"{args.save_masks}")
        args.save_masks.mkdir(parents=True, exist_ok=True)

    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)

    if args.verbose:
        print(f"main args: {args}")
        print(f"scripts config: {cfg}")

    sys.exit(main(args, cfg))
