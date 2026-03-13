# %%
# %load_ext autoreload
# %autoreload 2

import argparse
import os
import sys
import pathlib
from pathlib import Path

import numpy as np
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from mzbsuite.skeletonization.mzb_skeletons_pilmodel import MZBModel_skels
from mzbsuite.utils import cfg_to_arguments, SaveLogCallback

# Set the thread layer used by MKL
os.environ["MKL_THREADING_LAYER"] = "GNU"


def main(args, cfg):
    """
    Function to train a model for skeletons (body, head) on macrozoobenthos images.
    The model is trained on the dataset specified in the config file, and saved to 
    the folder specified in the config file every 50 steps and at the end of the training.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace containing the arguments passed to the script. Notably:
        
            - input_dir: path to the directory containing the images to be classified
            - save_model: path to the directory where the model will be saved
            - config_file: path to the config file with train / inference parameters

    cfg : dict
        Dictionary containing the configuration parameters.

    Returns
    -------
    None. Saves the model in the specified folder.
    """

    best_val_cb = pl.callbacks.ModelCheckpoint(
        dirpath=args.save_model,
        filename="best-val-{epoch}-{step}-{val_loss:.1f}",
        monitor="val_loss",
        mode="min",
        save_top_k=cfg.trsk_save_topk,
    )

    # latest model in training
    last_mod_cb = pl.callbacks.ModelCheckpoint(
        dirpath=args.save_model,
        filename="last-{step}",
        every_n_train_steps=50,
        save_top_k=cfg.trsk_save_topk,
    )

    # Define progress bar callback
    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

    # Define logger callback to log training date
    trdatelog = SaveLogCallback(model_folder=args.save_model)
        
    ### resolving Path in Windows
    if (sys.platform == "win32"):
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

    # Check if there is a model to load, if there is, load it and continue training
    if args.save_model.is_dir():
        if args.verbose:
            print(f"Loading model from {args.save_model}")
        
        # Find the checkpoint file
        try:
            fmodel = list(args.save_model.glob("last-*.ckpt"))[0]
        except IndexError:
            print("No last-* model in folder, loading best model")
            # Ensure we get the latest best model
            best_models = list(
                args.save_model.glob("best-val-epoch=*-step=*-val_loss=*.*.ckpt")
            )
            if not best_models:
                raise FileNotFoundError("No checkpoint files found in save directory.")
            fmodel = best_models[-1]
            
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model = MZBModel_skels(
        #     data_dir=args.input_dir,
        #     pretrained_network=cfg.trsk_model_pretrarch,  # .replace("-", "_"),
        #     learning_rate=cfg.trsk_learning_rate,
        #     batch_size=cfg.trsk_batch_size,
        #     weight_decay=cfg.trsk_weight_decay,
        #     num_workers_loader=cfg.trsk_num_workers,
        #     step_size_decay=cfg.trsk_step_size_decay,
        #     num_classes=cfg.trsk_num_classes,
        # )
       
        model = MZBModel_skels.load_from_checkpoint(
            checkpoint_path=fmodel,
            map_location=device,
            weights_only=False, # due to legacy checkpoint
        )
        
        # Override the data_dir with the command-line argument
        # The checkpoint might have saved a Linux path or an old path.
        model.data_dir = args.input_dir
        # Also update the internal folder references derived from data_dir
        model.im_folder = model.data_dir / "images"
        model.bo_folder = model.data_dir / "sk_body"
        model.he_folder = model.data_dir / "sk_head"

    name_run = f"skel-{model.architecture}"
    cbacks = [pbar_cb, best_val_cb, last_mod_cb, trdatelog]

    # Define logger, and use either wandb or tensorboard
    if cfg.trsk_logger == "wandb":
        logger = WandbLogger(
            project=cfg.trsk_wandb_project_name, name=name_run if name_run else None
        )
        logger.watch(model, log="all")

    elif cfg.trsk_logger == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=args.save_model,
            name=name_run if name_run else None,
            log_graph=True,
        )

    trainer = Trainer(
        accelerator="auto",  # cfg.trcl_num_gpus outdated
        max_epochs=cfg.trsk_number_epochs,
        # strategy=DDPStrategy(find_unused_parameters=False),
        precision="16-mixed",
        callbacks=cbacks,
        logger=logger,
        log_every_n_steps=1,
    )
    
    ### ======================================= ###
    # DEBUG: Check dataset info before training
    if args.verbose:
        print(f"\n=== DATASET DEBUG ===")
        print(f"Image folder: {model.im_folder}")
        print(f"Images found: {len(list(model.im_folder.glob('*.jpg')))}")
        print(f"Train indices: {len(model.trn_inds) if hasattr(model, 'trn_inds') else 'N/A'}")
        print(f"Val indices: {len(model.val_inds) if hasattr(model, 'val_inds') else 'N/A'}")
        # List files in input directory
        if args.input_dir.exists():
            files = list(args.input_dir.glob("**/*"))
            print(f"Files found in input_dir: {len(files)}")
            if files:
                print(f"First few files: {files[:5]}")
        else:
            print(f"WARNING: Input directory does not exist!")
        print(f"=====================\n")
    ### ======================================= ###

    trainer.fit(model)
    
    if (sys.platform == "win32"):
        pathlib.PosixPath = temp ### restore original pathlib function


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="path to config file with per-script args",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="path with images for training",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        required=True,
        help="path to where to save model checkpoints",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="print more info")
    args = parser.parse_args()

    with open(str(args.config_file), "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = cfg_to_arguments(cfg)

    if args.verbose:
        print(f"main args: {args}")
        print(f"scripts config: {cfg}")

    args.input_dir = Path(args.input_dir)
    args.save_model = Path(args.save_model)
    args.save_model = args.save_model / "checkpoints"

    np.random.seed(cfg.glob_random_seed)  # apply this seed to img transforms
    torch.manual_seed(cfg.glob_random_seed)  # needed for torchvision 0.7
    torch.cuda.manual_seed(cfg.glob_random_seed)  # needed for torchvision 0.7

    sys.exit(main(args, cfg))
