import sys
import shutil
import argparse
import yaml
import json
import re

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

from mzbsuite.utils import cfg_to_arguments

def natural_sort_key(path):
    """"
    Function to sort filename with multiple numbers throughout the name 
    by natural integers (i.e. 1, 2, 10) rather than lexicographical (i.e. 1, 10, 2). 
    """
    
    # Get the parent folder name (e.g., '0__1_b1_mixed_02_10_rgb')
    folder_name = path.parent.name
    
    # Find all numbers in the folder name
    numbers = re.findall(r'\d+', folder_name)
    
    if not numbers:
        return (0, str(path))
    
    # We want to sort primarily by the LAST number (the sample ID)
    # Convert the last number to an integer for correct numeric sorting
    last_num = int(numbers[-1])
    
    # Secondary sort: the full path string (to keep '02' before '01' if needed, 
    # or just to ensure deterministic order if IDs are identical)

    # Let's try sorting by the LAST number first, then the full path as a tiebreaker
    return (last_num, str(path))

def main(args, cfg):
    """
    Main function to prepare the manual skeleton annotations as prepared by phenopype (https://www.phenopype.org/). 
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
            - args.verbose: prints more info

    cfg: argparse.Namespace
        configuration options.

    Returns
    -------
    None. Everything is saved to disk or displayed on screen.
    """
    
    # ### ==================================== ###
    # # %% Manual args for debugging purposes
    # args = {}
    # args["config_file"] = f"{prefix}configs\mzb_example_config.yaml"
    # args[
    #     "input_raw_dir"
    # ] = "D:\phenopype\line_annotations"
    # args["input_clips_dir"] = "D:\phenopype\data"
    # args[
    #     "output_dir"
    # ] = f"{prefix}results\\mzb_example\\skeletons\\supervised_skeletons\\assessment"
    # args["skel_save_attributes"] = f"{prefix}\\results\\mzb_example\\skeletons\\supervised_skeletons\\assessment"
    # args["verbose"] = True
    # args = cfg_to_arguments(args)

    # with open(args.config_file, "r") as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # cfg = cfg_to_arguments(cfg)
    # ### ==================================== ###

    input_raw_dir = Path(args.input_raw_dir)
    input_clips_dir = Path(args.input_clips_dir)
    output_dir = Path(args.output_dir)

    # %% If any of the folders exist, interrupt the script and raise en error.
    if (output_dir).exists() and (
        (output_dir / "images")
        or (output_dir / "sk_body")
        or (output_dir / "sk_head")
    ):
        # print in red and then back to normal color
        raise ValueError(
            f"\033[91m{output_dir} already exists and contains data. Please delete or specify another folder.\033[0m"
        )
    else:
        output_dir.mkdir(exist_ok=True, parents=True)

    skel_save_attributes = Path(args.skel_save_attributes)
    skel_save_attributes.mkdir(exist_ok=True, parents=True)
    
    # define empty lists to store the data, columns and files to read
    measures = []
    cols = ["species", "clip_name", "head", "body"]
    files_to_merge = list(sorted(input_raw_dir.glob("**/*/*.json")))

    # %% loop over the files and extract the data, then append to the list, then merge, then save.
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
    
    # get clip based on the fact that it is an existing annotation in the folder
    annot_files = sorted(input_raw_dir.glob("**/*/annotations_v1.json"), key=natural_sort_key)
    
    (output_dir / "images").mkdir(exist_ok=True, parents=True)
    (output_dir / "sk_body").mkdir(exist_ok=True, parents=True)
    (output_dir / "sk_head").mkdir(exist_ok=True, parents=True)
    (skel_save_attributes / "plots").mkdir(exist_ok=True, parents=True)

    # %% Loop over the annotations, save the manual skeleton and optionally save plots.
    for file in annot_files:
        if args.verbose:
            print(f"\nProcessing: {file}")  # Debug: show which file
        
        gen_name = "_".join(file.parent.name.split("__")[1].split("_")[:-1])
        rgb_clip = gen_name + f"_rgb.{cfg.impa_image_format}"

        # Read the image and the annotation, to get the size of the image
        test_f_im = Path(input_clips_dir / rgb_clip)
        test_im = cv2.cvtColor(cv2.imread(str(test_f_im)), cv2.COLOR_BGR2RGB)
        
        # Load the annotations file directly       
        with open(file) as f:
            line = json.load(f)
        
        if args.verbose:
            # Debug: Check structure before accessing
            body = line.get('line', {}).get('body', {}).get('data', {}).get('line', [])
            head = line.get('line', {}).get('head', {}).get('data', {}).get('line', [])
            
            print(f"  Body entries: {len(body)}")
            print(f"  Head entries: {len(head)}")
            
        # Skip files with empty data
        if not body or not head:
            print(f"  ⚠️ Skipping - empty annotation data")
            continue
        
        # Copy the image to the output folder, only if annotations present
        shutil.copy(input_clips_dir / rgb_clip, output_dir / "images" / rgb_clip)
        
        # Get the polyline coordinates from the annotation file
        head = line['line']['head']['data']['line']
        body = line['line']['body']['data']['line']
        
        # Extract the first polyline (index 0) and convert to numpy array
        # CRITICAL: cast to int32 to ensure OpenCV compatibility
        body_coords = np.array(body[0], dtype=np.int32).reshape(-1, 1, 2)
        head_coords = np.array(head[0], dtype=np.int32).reshape(-1, 1, 2)
        
        if args.verbose: 
            # DEBUG: Verify the image path matches the annotation context
            print(f"  Loading image: {test_f_im}")
            if test_im is None:
                print(f"  ERROR: Could not load image {test_f_im}")
                continue

            # Double check bounds with explicit values
            print(f"  Body coords min/max: {body_coords.min()}, {body_coords.max()}")
            print(f"  Head coords min/max: {head_coords.min()}, {head_coords.max()}")
            print(f"  Image dims: H={test_im.shape[0]}, W={test_im.shape[1]}")

            # Create mask
            mask_body = np.zeros_like(test_im, dtype=np.uint8)
            
            # Check that polylines actually changes pixel values in mask
            result = cv2.polylines(mask_body, [body_coords], isClosed=False, color=(0, 255, 0), thickness=30)
            
            # Check result (polylines returns the modified image, but modifies in place too)
            print(f"  Non-zero pixels after draw: {np.count_nonzero(mask_body)}")
            
            # If still 0, try a sanity check: draw a simple square
            if np.count_nonzero(mask_body) == 0:
                print("  ⚠️ Drawing failed. Trying sanity check (drawing a square)...")
                cv2.rectangle(mask_body, (10, 10), (50, 50), (0, 255, 0), -1)
                print(f"  Sanity check non-zero: {np.count_nonzero(mask_body)}")
                # If sanity check works, the issue is definitely the coordinates or type.

        ### Actually drawing on the masks
        # Draw the lines corresponding to body size on the image and save
        bw_mask = np.zeros_like(test_im)
        body_img = cv2.polylines(
            bw_mask,
            # np.array(body[0]).reshape(-1, 1, 2),
            [body_coords],
            isClosed=False,
            color=(0, 255, 0),
            thickness=cfg.skel_label_thickness,
        )
        cv2.imwrite(str(output_dir / "sk_body" / f"{gen_name}_body_skel.jpg"), body_img)

        # Draw the lines corresponding to head size on the image and save
        bw_mask = np.zeros_like(test_im)
        head_img = cv2.polylines(
            bw_mask,
            # np.array(head[0]).reshape(-1, 1, 2),
            [head_coords],
            isClosed=False,
            color=(0, 0, 255),
            thickness=cfg.skel_label_thickness,
        )
        cv2.imwrite(str(output_dir / "sk_head" / f"{gen_name}_head_skel.jpg"), head_img)
        
        ### Comparison plots to check drawing seems sensible
        # Extract the actual coordinate arrays (remove the extra nesting level)
        body_coords = np.array(body[0])  # Shape: (N, 2)
        head_coords = np.array(head[0])  # Shape: (M, 2)

        # Create a figure with 3 subplots side by side
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Original RGB image
        axes[0].imshow(test_im)
        axes[0].set_title('Original RGB Clip', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Plot 2: Body skeleton overlay
        axes[1].imshow(test_im)  # Show original as background
        axes[1].plot(body_coords[:, 0], body_coords[:, 1], 
                    color='green', linewidth=cfg.skel_label_thickness, label='Body')
        axes[1].scatter(body_coords[:, 0], body_coords[:, 1], 
                        c='green', s=50, marker='o')  # Mark points
        axes[1].set_title('Body Length Annotation', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper right')
        axes[1].axis('off')

        # Plot 3: Head skeleton overlay
        axes[2].imshow(test_im)  # Show original as background
        axes[2].plot(head_coords[:, 0], head_coords[:, 1], 
                    color='red', linewidth=cfg.skel_label_thickness, label='Head')
        axes[2].scatter(head_coords[:, 0], head_coords[:, 1], 
                        c='red', s=50, marker='o')  # Mark points
        axes[2].set_title('Head Width Annotation', fontsize=12, fontweight='bold')
        axes[2].legend(loc='upper right')
        axes[2].axis('off')
        
        # Adjust layout and display
        plt.tight_layout()
        
        # Save comparison plot
        save_path = skel_save_attributes / "plots" / f"{gen_name}_comparison.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if args.verbose: 
            print(f"Saved comparison to: {save_path}")
        
        if PLOTS:            
            plt.show()
        
        plt.close()
    
    
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
