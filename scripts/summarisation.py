import os
import argparse
import pandas as pd


def find_csv_in_folder(folder, pattern):
    # If only one subfolder, go inside it
    entries = [os.path.join(folder, e) for e in os.listdir(folder)]
    dirs = [e for e in entries if os.path.isdir(e)]
    if len(dirs) == 1:
        folder = dirs[0]
    for fname in os.listdir(folder):
        if fname.endswith('.csv') and pattern in fname:
            return os.path.join(folder, fname)
    raise FileNotFoundError(f"No CSV file matching '{pattern}' in {folder}")


def strip_extension(s):
    if pd.isnull(s):
        return s
    return os.path.splitext(str(s))[0]


def main():
    parser = argparse.ArgumentParser(description='Merge classification, size_skel, and skeleton_attributes CSVs.')
    parser.add_argument('--classification', required=True, help='Folder containing classification CSV')
    parser.add_argument('--skeletons_supervised', required=True, help='Folder containing size_skel_supervised_model.csv')
    parser.add_argument('--skeletons_unsupervised', required=True, help='Folder containing skeleton_attributes.csv')
    parser.add_argument('--output_folder', required=True, help='Folder to save merged output')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')
    args = parser.parse_args()

    # Find files
    class_path = find_csv_in_folder(args.classification, 'classification_predictions')
    size_path = find_csv_in_folder(args.skeletons_supervised, 'size_skel_supervised_model')
    skel_path = find_csv_in_folder(args.skeletons_unsupervised, 'skeleton_attributes')

    if args.verbose:
        print(f"Classification file: {class_path}")
        print(f"Size skel file: {size_path}")
        print(f"Skeleton attributes file: {skel_path}")

    # Read CSVs
    df_class = pd.read_csv(class_path)
    df_size = pd.read_csv(size_path)
    df_skel = pd.read_csv(skel_path)

    # Strip extensions from merge columns
    if 'file' in df_class.columns:
        df_class['file_noext'] = df_class['file'].apply(strip_extension)
    if 'clip_name' in df_size.columns:
        df_size['clip_name_noext'] = df_size['clip_name'].apply(strip_extension)
    if 'clip_filename' in df_skel.columns:
        df_skel['clip_filename_noext'] = df_skel['clip_filename'].apply(strip_extension)

    if args.verbose:
        print(f"Classification shape: {df_class.shape}")
        print(f"Size skel shape: {df_size.shape}")
        print(f"Skeleton attributes shape: {df_skel.shape}")

    # Merge on extension-stripped columns
    df_merged = pd.merge(df_class, df_size, left_on='file_noext', right_on='clip_name_noext', how='outer')
    df_merged = pd.merge(df_merged, df_skel, left_on='file_noext', right_on='clip_filename_noext', how='outer')

    if args.verbose:
        print(f"Merged shape: {df_merged.shape}")
        print(f"Columns: {df_merged.columns.tolist()}")

    # Output
    os.makedirs(args.output_folder, exist_ok=True)
    out_path = os.path.join(args.output_folder, 'merged_output.csv')
    df_merged.to_csv(out_path, index=False)
    print(f"Merged file saved as {out_path}")


if __name__ == '__main__':
    main()