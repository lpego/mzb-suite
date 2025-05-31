import os, sys
import argparse
import pandas as pd

import yaml
from mzbsuite.utils import cfg_to_arguments

def find_csv_in_folder(folder, pattern):
    import re
    from datetime import datetime
    # If only one subfolder, go inside it
    entries = [os.path.join(folder, e) for e in os.listdir(folder)]
    dirs = [e for e in entries if os.path.isdir(e)]
    if len(dirs) == 1:
        folder = dirs[0]
    elif len(dirs) > 1:
        # Try to match datetime pattern in folder names and pick the most recent
        dt_pattern = re.compile(r'(\d{8}_\d{4})')
        dt_folders = []
        for d in dirs:
            m = dt_pattern.search(os.path.basename(d))
            if m:
                try:
                    dt = datetime.strptime(m.group(1), '%Y%m%d_%H%M')
                    dt_folders.append((dt, d))
                except Exception:
                    pass
        if dt_folders:
            dt_folders.sort(reverse=True)
            folder = dt_folders[0][1]
            print(f"Warning: Multiple folders found in {os.path.abspath(os.path.dirname(folder))}. Using most recent: {os.path.basename(folder)}")
    for fname in os.listdir(folder):
        if fname.endswith('.csv') and pattern in fname:
            return os.path.join(folder, fname)
    raise FileNotFoundError(f"No CSV file matching '{pattern}' in {folder}")


def strip_extension(s):
    if pd.isnull(s):
        return s
    base = os.path.splitext(str(s))[0]
    # Remove trailing _rgb or _mask if present
    for suffix in ['_rgb', '_mask']:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def main(args, cfg):
    """
    Function to summarse classification, unsupervised skeletons and supervised skeletons into a single CSV.
    Parameters

    ----------
    args : argparse.Namespace
        - classification: path to the folder containing classification_predictions.csv
        - skeletons_supervised: path to the folder containing supervised_skeletons.csv
        - skeletons_unsupervised: path to the folder containing unsupervised_skeletons.csv
         - taxonomy_file: path to the taxonomy file used for the classification (optional)
        - output_folder: path to the folder where the merged output will be saved
        - verbose: whether to print verbose output
    cfg : dict
        - config_file: path to the config file with per-script args (required if taxonomy file provided)

    Returns
    -------
    None. Saves the results in the specified folder.
    """

    # Find files
    class_path = find_csv_in_folder(args.classification, 'classification_predictions')
    skel_unsup_path = find_csv_in_folder(args.skeletons_unsupervised, 'unsupervised_skeletons')
    skel_sup_path = find_csv_in_folder(args.skeletons_supervised, 'supervised_skeletons')

    if args.verbose:
        print(f"Classification file: {class_path}")
        print(f"Unsupervised skeletons file: {skel_unsup_path}")
        print(f"Supervised skeletons file: {skel_sup_path}")

    # Read CSVs
    df_class = pd.read_csv(class_path)
    df_skel_unsup = pd.read_csv(skel_unsup_path)
    df_skel_sup = pd.read_csv(skel_sup_path)

    # Strip extensions from merge columns
    if 'file' in df_class.columns:
        df_class['file_noext'] = df_class['file'].apply(strip_extension)
        df_class.drop(columns=['file'], inplace=True)
    if 'clip_filename' in df_skel_unsup.columns:
        df_skel_unsup['clip_name_noext'] = df_skel_unsup['clip_filename'].apply(strip_extension)
        df_skel_unsup.drop(columns=['clip_filename'], inplace=True)
    if 'clip_name' in df_skel_sup.columns:
        df_skel_sup['clip_filename_noext'] = df_skel_sup['clip_name'].apply(strip_extension)
        df_skel_sup.drop(columns=['clip_name'], inplace=True)

    if args.verbose:
        print(f"Classification shape: {df_class.shape}")
        print(f"Unsupervised skeletons shape: {df_skel_unsup.shape}")
        print(f"Skeleton attributes shape: {df_skel_sup.shape}")
        # Print first 10 rows of zipped merge columns
        zipped = list(zip(
            df_class['file_noext'] if 'file_noext' in df_class else [],
            df_skel_unsup['clip_name_noext'] if 'clip_name_noext' in df_skel_unsup else [],
            df_skel_sup['clip_filename_noext'] if 'clip_filename_noext' in df_skel_sup else []
        ))
        print("First 10 rows of merge columns (file_noext, clip_name_noext, clip_filename_noext):")
        for row in zipped[:10]:
            print(row)

    # Merge on extension-stripped columns, avoid duplicate key columns
    df_merged = pd.merge(
        df_class,
        df_skel_unsup,
        left_on='file_noext',
        right_on='clip_name_noext',
        how='outer',
        suffixes=(None, '_unsup')
    )
    df_merged = pd.merge(
        df_merged,
        df_skel_sup,
        left_on='file_noext',
        right_on='clip_filename_noext',
        how='outer',
        suffixes=(None, '_sup')
    )
    # Drop duplicate merge columns if present
    for col in ['clip_name_noext', 'clip_filename_noext']:
        if col in df_merged.columns:
            df_merged.drop(columns=[col], inplace=True)

    if args.verbose:
        print(f"Merged shape: {df_merged.shape}")
        print(f"Columns: {df_merged.columns.tolist()}")
        
    # If taxonomy file is provided, map class numbers to names in 'pred' column
    if args.taxonomy_file is not None:
        print("Taxonomy file found, proceeding to map class numbers to names.")
        try:
            mzb_taxonomy = pd.read_csv(args.taxonomy_file)
            mzb_taxonomy = mzb_taxonomy.drop(columns=["Unnamed: 0"])
            mzb_taxonomy = mzb_taxonomy.ffill(axis=1)
            # watch out this sorted is important for the class names to be in the right order
            class_names = sorted(
                list(mzb_taxonomy[cfg.lset_class_cut].str.lower().unique())
            )
        except Exception as e:
            print(f"[ERROR] Could not parse config with cfg_to_arguments or missing lset_class_cut: {e}")
            
        # Build mapping: class number (row index starting at 0) -> class name (lowercase)
        class_map = {i: name for i, name in enumerate(class_names)}
        if 'pred' in df_merged.columns:
            df_merged['pred_class'] = df_merged['pred'].map(class_map)
            if args.verbose:
                print('Added pred_class column with class names mapped from provided taxonomy file (index starting at 1).')

    # Output
    # Reorder columns as requested
    preferred_order = [
        'file_noext',
        'pred_class',
        'skel_length',
        'skel_length_mm',
        'conv_rate_mm_px',
        'segms',
        'area',
        'nn_pred_body',
        'nn_pred_head'
    ]
    # Add any remaining columns not in preferred_order
    rest = [col for col in df_merged.columns if col not in preferred_order]
    final_order = [col for col in preferred_order if col in df_merged.columns] + rest
    df_merged = df_merged[final_order]

    os.makedirs(args.output_folder, exist_ok=True)
    out_path = os.path.join(args.output_folder, 'merged_output.csv')
    df_merged.to_csv(out_path, index=False)
    print(f"Merged file saved as {out_path}")


if __name__ == '__main__':
     # # --- MANUAL ARGS BLOCK FOR NOTEBOOK OR SCRIPT TESTING ---
    # # Uncomment and edit the following lines to override argparse for quick testing:
    # class Args:
    #     classification = r'D:\mzb-workflow\results\swiss-invertebrates\classification'
    #     skeletons_supervised = r'D:\mzb-workflow\results\swiss-invertebrates\skeletons\supervised_skeletons'
    #     skeletons_unsupervised = r'D:\mzb-workflow\results\swiss-invertebrates\skeletons\unsupervised_skeletons'
    #     output_folder = r'D:\mzb-workflow\results\swiss-invertebrates'
    #     config_file = r'D:\mzb-workflow\configs\mzb_example_config.yaml'
    #     verbose = True
    # args = Args()
    
    # args = parser.parse_args()
    # with open(str(args.config_file), "r") as f: 
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # cfg = cfg_to_arguments(cfg)
    # # --------------------------------------------------------

    parser = argparse.ArgumentParser(description='Merge classification, unsupervised skeletons and supervised skeleton into single CSV.')
    parser.add_argument('--classification', required=True, help='Folder containing classification_predictions.csv')
    parser.add_argument('--skeletons_supervised', required=True, help='Folder containing supervised_skeletons.csv')
    parser.add_argument('--skeletons_unsupervised', required=True, help='Folder containing unsupervised_skeletons.csv')
    parser.add_argument('--taxonomy_file', required=False, help='Path to the taxonomy file used for the classification')
    parser.add_argument('--output_folder', required=True, help='Folder to save merged output')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')
    parser.add_argument("--config_file", required=False, help="path to config file with per-script args")
    # # Only parse args if not manually set above
    # if 'args' not in locals():
    #     args = parser.parse_args()
        
    #     with open(str(args.config_file), "r") as f:
    #         cfg = yaml.load(f, Loader=yaml.FullLoader)

    #     cfg = cfg_to_arguments(cfg)
    args = parser.parse_args()

    with open(str(args.config_file), "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = cfg_to_arguments(cfg)

    sys.exit(main(args, cfg))
