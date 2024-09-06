# %%
# This script is not working as the filename have been changing in the new pipeline derived files.
# The actual skeleton learning sets are stable, as those were copied over, but we might think about redoing this script in the future.

import sys
import shutil
import argparse
import yaml
import json

import cv2
import numpy as np
import pandas as pd

from pathlib import Path

from matplotlib import pyplot as plt

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
    PLOTS = False
else:
    prefix = "../../"  # or "../"
    PLOTS = True

sys.path.append(f"{prefix}")

from mzbsuite.utils import cfg_to_arguments  # , noneparse

def main(args, cfg):
    """
    Main function to prepare the manual annotations as prepared by phenopype (https://www.phenopype.org/). 
    Collects and reorganises in a single file head width and body length measurements. 
    
    Parameters
    ----------
    args: argparse.Namespace
        Arguments parsed from the command line. Specifically:
            - config_file: path to the configuration file
            - args.input_raw_dir: path to the directory with the manual annotations
            - args.input_clips_dir: path to the directory with the clips the annotations were measured on
            - args.skel_save_attributes: path to where the summarised annotations are going to be saved
            - args.output_dir: path where the clips with the skeletons superimposed are going to be saved
            - args.verbose: prints more info (currently does nothing)

    cfg: argparse.Namespace
        configuration options.

    Returns
    -------
    None. Everything is saved to disk or displayed on screen.
    """
    
    # args = {}
    # args["config_file"] = f"{prefix}configs/global_configuration.yaml"
    # args[
    #     "input_raw_dir"
    # ] = f"{prefix}data/raw/2021_swiss_invertebrates/manual_measurements/"
    # args["input_clips_dir"] = f"{prefix}/data/derived/project_portable_flume/blobs/"
    # args[
    #     "output_dir"
    # ] = f"{prefix}data/learning_sets/project_portable_flume/skeletonization/"
    # args["verbose"] = True
    # args = cfg_to_arguments(args)

    # with open(args.config_file, "r") as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # cfg = cfg_to_arguments(cfg)

    input_raw_dir = Path(args.input_raw_dir)
    input_clips_dir = Path(args.input_clips_dir)
    output_dir = Path(args.output_dir)

    # if any of the folders exist, interrupt the script and raise en error.
    if (output_dir).exists() and (
        (output_dir / "images")
        or (output_dir / "sk_body")
        or (output_dir / "sk_head")
    ):
        # print in red and then back to normal color
        raise ValueError(
            f"\033[91m{output_dir} already exists and contains data. Please delete or sprecify another folder.\033[0m"
        )
    else:
        output_dir.mkdir(exist_ok=True, parents=True)

    skel_save_attributes = Path(args.skel_save_attributes)
    skel_save_attributes.mkdir(exist_ok=True, parents=True)
    # %%
    # define empty lists to store the data, columns and files to read
    measures = []
    cols = ["species", "clip_name", "head", "body"]
    files_to_merge = list(sorted(input_raw_dir.glob("**/*/*.json")))

    # loop over the files and extract the data, then append to the list, then merge, then save.
    for jfi in files_to_merge[:]:
        clip_name = jfi.parent.name.split("__")[-1]
        species = clip_name.split("_")[2]

        with open(jfi) as f:
            data = json.load(f)

        body = data["line"]["body"]["data"]["lengths"]
        head = data["line"]["head"]["data"]["lengths"]

        measures.append(
            pd.DataFrame(
                {
                    "clip_name": clip_name,
                    "species": species,
                    "head_length": head,
                    "body_length": body,
                }
            )
        )

    all_measures = pd.concat(measures)
    all_measures.to_csv(
        skel_save_attributes / "manual_annotations_summary.csv", index=False
    )
    # %%
    # Get clip based on the fact that it is an existing annotation in Danina's folder
    annot_files = sorted(list(input_raw_dir.glob("**/*/annotations_v1.json")))

    (output_dir / "images").mkdir(exist_ok=True, parents=True)
    (output_dir / "sk_body").mkdir(exist_ok=True, parents=True)
    (output_dir / "sk_head").mkdir(exist_ok=True, parents=True)

    # %%
    # Loop over the annotations and copy the image and save the manual skeleton.
    for file in annot_files:
        gen_name = "_".join(file.parent.name.split("__")[1].split("_")[:-1])
        rgb_clip = gen_name + f"_rgb.{cfg.impa_image_format}"

        # Copy the image to the output folder
        shutil.copy(input_clips_dir / rgb_clip, output_dir / "images")

        # Read the image and the annotation, to get the size of the image
        test_f_im = Path(input_clips_dir / rgb_clip)
        test_im = cv2.cvtColor(cv2.imread(str(test_f_im)), cv2.COLOR_BGR2RGB)

        if PLOTS:
            plt.figure()
            plt.imshow(test_im)

        # Read the annotation file and get the head and body line coordinates
        # line = pd.read_csv(file)
        # body = line[["x_coords", "y_coords"]].loc[line.annotation_id == "body"].values
        # head = line[["x_coords", "y_coords"]].loc[line.annotation_id == "head"].values
        
        with open(annot_files[0]) as f:
            line = json.load(f)
            
        head = line['line']['head']['data']['line']
        body = line['line']['body']['data']['line']

        # Draw the lines corresponding to body size on the image and save
        bw_mask = np.zeros_like(test_im)
        body_img = cv2.polylines(
            bw_mask,
            np.array(body), # no need to unpack the list
            isClosed=False,
            color=(0, 255, 0),
            thickness=cfg.skel_label_thickness,
        )
        cv2.imwrite(str(output_dir / "sk_body" / f"{gen_name}_body_skel.png"), body_img)

        if PLOTS:
            plt.figure()
            plt.imshow(body_img)

        # Draw the lines corresponding to head size on the image and save
        bw_mask = np.zeros_like(test_im)
        head_img = cv2.polylines(
            bw_mask,
            np.array(head),
            isClosed=False,
            color=(255, 0, 0),
            thickness=cfg.skel_label_thickness,
        )
        cv2.imwrite(str(output_dir / "sk_head" / f"{gen_name}_head_skel.png"), head_img)

        if PLOTS:
            plt.figure()
            plt.imshow(head)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--input_raw_dir", type=str, required=True)
    parser.add_argument("--input_clips_dir", type=str, required=True)
    parser.add_argument("--skel_save_attributes", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = cfg_to_arguments(cfg)

    sys.exit(main(args, cfg))
