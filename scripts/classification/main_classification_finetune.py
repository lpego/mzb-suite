# %%
# %load_ext autoreload
# %autoreload 2

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
else:
    prefix = "../../"  # or "../"

sys.path.append(f"{prefix}")

from mzbsuite.classification.mzb_classification_pilmodel import MZBModel
from mzbsuite.utils import cfg_to_arguments, SaveLogCallback

# Set the thread layer used by MKL
os.environ["MKL_THREADING_LAYER"] = "GNU"


# %%
def main(args, cfg):
    """
    Function to train a model for classification of macrozoobenthos images.
    The model is trained on the dataset specified in the config file.
    The model is saved in the folder specified in the config file.
    The model is saved every 50 steps and at the end of the training.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace containing the arguments passed to the script.
    cfg : dict
        Dictionary containing the configuration parameters.

    Returns
    -------
    None. Saves the model in the specified folder.
    """

    # Define checkpoints callbacks
    # best model on validation
    best_val_cb = pl.callbacks.ModelCheckpoint(
        dirpath=args.save_model,
        filename="best-val-{epoch}-{step}-{val_loss:.1f}",
        monitor="val_loss",
        mode="min",
        save_top_k=cfg.trcl_save_topk,
    )

    # latest model in training
    last_mod_cb = pl.callbacks.ModelCheckpoint(
        dirpath=args.save_model,
        filename="last-{step}",
        every_n_train_steps=50,
        save_top_k=cfg.trcl_save_topk,
    )

    # Define progress bar callback
    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

    # Define logger callback to log training date
    trdatelog = SaveLogCallback(model_folder=args.save_model)

    # Define model from config
    model = MZBModel(
        data_dir=args.input_dir,
        pretrained_network=cfg.trcl_model_pretrarch,
        learning_rate=cfg.trcl_learning_rate,
        batch_size=cfg.trcl_batch_size,
        weight_decay=cfg.trcl_weight_decay,
        num_workers_loader=cfg.trcl_num_workers,
        step_size_decay=cfg.trcl_step_size_decay,
        num_classes=cfg.trcl_num_classes,
    )

    # Check if there is a model to load, if there is, load it and train from there
    if args.save_model.is_dir():
        if args.verbose:
            print(f"Loading model from {args.save_model}")
        try:
            fmodel = list(args.save_model.glob("last-*.ckpt"))[0]
        except:
            print("No last-* model in folder, loading best model")
            fmodel = list(
                args.save_model.glob("best-val-epoch=*-step=*-val_loss=*.*.ckpt")
            )[-1]

        model = model.load_from_checkpoint(fmodel)

    # Define logger and name of run
    name_run = f"classifier-{cfg.trcl_model_pretrarch}"  # f"{model.pretrained_network}"
    cbacks = [pbar_cb, best_val_cb, last_mod_cb, trdatelog]

    # TODOs: add possibility to use tensorboard
    wb_logger = WandbLogger(
        project=cfg.trcl_wandb_project_name, name=name_run if name_run else None
    )
    wb_logger.watch(model, log="all")

    # instantiate trainer and train
    trainer = Trainer(
        gpus=cfg.trcl_gpu_ids,  # [0,1],
        max_epochs=cfg.trcl_number_epochs,
        strategy=DDPStrategy(
            find_unused_parameters=False
        ),  # TODO: check how to use in notebook
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

    # args = {}
    # args["config_file"] = f"{prefix}configs/global_configuration.yaml"
    # args[
    #     "input_dir"
    # ] = f"{prefix}data/learning_sets/project_portable_flume/aggregated_learning_sets/"
    # args["save_model"] = f"{prefix}models/mzb-class/"
    # args["verbose"] = True
    # args = cfg_to_arguments(args)

    with open(str(args.config_file), "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = cfg_to_arguments(cfg)

    if args.verbose:
        print(f"main args: {args}")
        print(f"scripts config: {cfg}")

    args.input_dir = Path(args.input_dir)
    args.save_model = Path(args.save_model)
    args.save_model = args.save_model / "checkpoints"

    np.random.seed(cfg.glob_random_seed)  # apply this seed to img tranfsorms
    torch.manual_seed(cfg.glob_random_seed)  # needed for torchvision 0.7
    torch.cuda.manual_seed(cfg.glob_random_seed)  # needed for torchvision 0.7

    main(args, cfg)
