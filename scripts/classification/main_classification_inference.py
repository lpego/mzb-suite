# %%
# this script is used to perform inference on a trained model, on a folder containing images.
# The structure of the folder is expected to be:
# - input_dir
#   - class1
#     - image1
#     - image2
#     - ...
#   - class2
#     - image1
#     - image2
#     - ...
#   - ...
#
# Alternatively, if no class structure is present, all images can be in a single folder, and the model will predict the class for each image.
#
# The output is a csv file with the following columns:
# - image_path
# - predicted class
# - probability for each class
# - true class (if available, eg when the input folder is structured as above)
#

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml
import torch

# try:
#     __IPYTHON__
# except:
#     prefix = ""  # or "../"
# else:
#     prefix = "../../"  # or "../"

# sys.path.append(f"{prefix}")

from mzbsuite.classification.mzb_classification_pilmodel import MZBModel
from mzbsuite.utils import cfg_to_arguments, find_checkpoints

# Set the thread layer used by MKL
os.environ["MKL_THREADING_LAYER"] = "GNU"

# %%


# %%
def main(args, cfg):
    """
    TODO: add docstring
    """
    dirs = find_checkpoints(
        Path(args.input_model).parents[0],
        version=Path(args.input_model).name,
        log=cfg.infe_model_ckpt,
    )

    mod_path = dirs[0]

    # ckpt = torch.load(mod_path, map_location=torch.device("cpu"))
    # hprs = ckpt["hyper_parameters"]

    # model = MZBModel(
    #     data_dir=hprs["data_dir"],
    #     pretrained_network=hprs["pretrained_network"],
    #     learning_rate=hprs["learning_rate"],
    #     batch_size=hprs["batch_size"],
    #     weight_decay=hprs["weight_decay"],
    #     num_workers_loader=hprs["num_workers_loader"],
    #     step_size_decay=hprs["step_size_decay"],
    # )

    model = MZBModel()
    model = model.load_from_checkpoint(
        checkpoint_path=mod_path,
    )

    model.data_dir = Path(args.input_dir)
    model.num_classes = cfg.infe_num_classes

    model.eval()
    # %%
    # dataloader = model.train_dataloader(shuffle=False)
    # dataloader = model.dubendorf_dataloader()
    if "val_set" in model.data_dir.name:
        dataloader = model.val_dataloader()
    else:
        dataloader = model.external_dataloader(
            model.data_dir, glob_pattern=cfg.infe_image_glob
        )

    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        callbacks=[pbar_cb],
        enable_checkpointing=False,
        logger=False,
    )

    outs = trainer.predict(
        model=model, dataloaders=[dataloader], return_predictions=True
    )

    # %%
    if cfg.lset_taxonomy:
        mzb_taxonomy = pd.read_csv(Path(cfg.lset_taxonomy))
        mzb_taxonomy = mzb_taxonomy.drop(columns=["Unnamed: 0"])
        mzb_taxonomy = mzb_taxonomy.ffill(axis=1)
        # watch out this sorted is important for the class names to be in the right order
        class_names = sorted(
            list(mzb_taxonomy[cfg.lset_class_cut].str.lower().unique())
        )

    # %%
    y = []
    p = []
    gt = []
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

    # make now output csv containing the file name, the class and the probabilities of prediction.
    # if available, also add the ground truth class
    data = {
        "file": [f.name for f in dataloader.dataset.img_paths],
        "pred": np.argmax(pc, axis=1),
    }

    for clanam in class_names:
        data[clanam] = pc[:, class_names.index(clanam)]

    if "val_set" in model.data_dir.name:
        data["gt"] = gc
    else:
        data["gt"] = 0

    out_dir = (
        Path(args.output_dir)
        / f"{model.data_dir.name}_{Path(args.input_model).name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    csv_name = f"predictions.csv"
    df_ = pd.DataFrame(data)
    df_.to_csv(out_dir / csv_name, index=False)

    # %%
    if "val_set" in model.data_dir.name:
        from matplotlib import pyplot as plt
        from sklearn.metrics import (
            confusion_matrix,
            ConfusionMatrixDisplay,
            classification_report,
        )

        cmat = confusion_matrix(gc, np.argmax(pc, axis=1), normalize="true")
        f = plt.figure(figsize=(10, 10))
        aa = f.gca()
        cm_disp = ConfusionMatrixDisplay.from_predictions(
            gc,
            yc,
            ax=aa,
            values_format=".1f",
            normalize=None,
            xticks_rotation="vertical",
            cmap="Greys",
            display_labels=class_names,
        )
        plt.savefig(out_dir / "confusion_matrix.png", dpi=300)

        rep_txt = classification_report(
            gc, np.argmax(pc, axis=1), target_names=class_names
        )
        with open(out_dir / "classification_report.txt", "w") as f:
            f.write(rep_txt)


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
        help="path with images to perform inference on",
    )
    parser.add_argument(
        "--input_model",
        type=str,
        required=True,
        help="path to model checkpoint for inference",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to where to save classificaiton predictions as csv",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="print more info")
    args = parser.parse_args()

    with open(str(args.config_file), "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = cfg_to_arguments(cfg)

    main(args, cfg)

# %%
if 0:
    import torch
    from matplotlib import pyplot as plt
    from PIL import Image

    from mzb_workflow.classification.mzb_classification_dataloader import Denormalize

    dd = "results/classification/project_portable_flume/mixed_set_convnext-small-v0_20230309_1737/predictions.csv"
    df_pred = pd.read_csv(prefix + dd)
    df_pred = df_pred.set_index("file")
    df_pred = df_pred.sort_index()

    mzb_class = 4

    denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    unc_score = -pc[:, mzb_class]

    # # unc_score = np.sum(pc * np.log(pc), axis=1)
    # # unc_score = np.sort(pc,axis=1)[:,-1] - np.sort(pc,axis=1)[:,-2]

    # ss = np.argsort(-pc[:, mzb_class])
    ss = np.argsort(unc_score)
    sub = ss
    # sub = ss[gc[ss] == mzb_class]

    files = dataloader.dataset.img_paths

    plt.figure(figsize=(15, 5))
    for c, name in enumerate(class_names):
        # plt.hlines(gc[gc==c], xmin=np.where(gc==c)[0][0], xmax=np.where(gc==c)[0][-1])
        plt.scatter(
            np.where(gc == c)[0], np.ones_like(gc[gc == c]), label=f"{c}: {name}"
        )
    plt.plot(pc)
    plt.legend()
    plt.figure(figsize=(15, 5))
    plt.plot(pc[yc == mzb_class, :])
    plt.legend()

    print(f"PREDICTING CLASS {class_names[mzb_class]}")

    DETECT = 0
    DIFF = True
    FR = 0
    N = 10

    preds = []

    for i, ti in enumerate(sub):
        if i < FR:
            continue

        fi = files[ti]
        im = Image.open(fi).convert("RGB")
        x = model.transform_ts(im)
        x = x[np.newaxis, ...]

        with torch.set_grad_enabled(False):
            p = torch.softmax(model(x), dim=1).cpu().numpy()
            pl_im = denorm(np.transpose(x.squeeze(), (1, 2, 0)))

        f, a = plt.subplots(1, 1, figsize=(4, 4))
        p_class = np.argmax(pc[ti, :])

        a.imshow(pl_im)
        a.axis("off")
        a.set_title(
            f"Inference: predicted class {p_class} with P {pc[ti, p_class]:.2f}\n"
            f"{files[ti]} \n GT {gc[ti]},  "
            f"Y @ {pc[ti,mzb_class]:.2f}"
        )

        # if pc[ti, mzb_class] < 0.01:
        # break

        if i > N:
            break

# %%
