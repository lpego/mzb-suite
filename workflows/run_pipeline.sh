#!/bin/bash 

ROOT_DIR="/data/shared/mzb-workflow"

# python scripts/image_parsing/main_raw_to_clips.py \
#     --input_dir=$ROOT_DIR/data/raw/project_portable_flume \
#     --output_dir=$ROOT_DIR/data/results/project_portable_flume/blobs/ \
#     --save_full_mask_dir=$ROOT_DIR/data/results/project_portable_flume/full_image_masks \
#     --config_file=$ROOT_DIR/configs/global_configuration.yaml \
#     -v

## CHECK: _mask_properties.csv file in the output directory, does not get filled correctly  

# This has to be run once, to create the curated learning sets
# python scripts/classification/main_prepare_learning_sets.py \
#     --input_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/curated_learning_sets_flume/ \
#     --taxonomy_file=$ROOT_DIR/data/MZB_taxonomy.csv \
#     --output_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/aggregated_learning_sets \
#     --config_file=$ROOT_DIR/configs/global_configuration.yaml \
#     -v


# This is run to classify a custom folder structure and will regturn a csv with the results 
# (filename, predicted class, probability of prediction)
# eg. on a validaton / test set, or on a new set of images, or whatever. be assured to pass only clips generated similarty 
# to the first step (main_raw_to_clips.py)
# classification models are stored in $ROOT_DIR/models/mzb-pil/
# python scripts/classification/main_classification_inference.py \
#     --config_file=$ROOT_DIR/configs/global_configuration.yaml \
#     --input_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/aggregated_learning_sets/val_set \
#     --input_model=$ROOT_DIR/models/mzb-pil/bm2ccwxc \
#     --output_dir=$ROOT_DIR/results/classification/project_portable_flume/ \
#     -v

### Ex. run on all the clips avaiable to get full predictions
## python scripts/classification/main_classification_inference.py \
##     --config_file=$ROOT_DIR/configs/global_configuration.yaml \
##     --input_dir=$ROOT_DIR/data/derived/project_portable_flume/blobs/ \
##     --input_model=$ROOT_DIR/models/mzb-pil/bm2ccwxc \
##     --output_dir=$ROOT_DIR/results/classification/project_portable_flume/ \
##     -v

# This part run the unsupervised skeletonization and measurement. It will read all the mask clips created at the first step
# and will return a csv with the results (filename, skeleton, etc)
# The pipeline to get these numbers is unsupervised, but can only approximate length of the insect, and not the width of the head.
# For this, the next step that uses a supervised neural network is required.

# python scripts/skeletons/main_unsupervised_skeleton_estimation.py \
#     --config_file=$ROOT_DIR/configs/global_configuration.yaml \
#     --input_dir=$ROOT_DIR/data/derived/project_portable_flume/blobs/ \
#     --output_dir=$ROOT_DIR/results/skeletons/project_portable_flume \
#     --list_of_files=None \
#     -v

# python scripts/skeletons/main_skeleton_supervised_inference.py \
#     --config_file=$ROOT_DIR/configs/global_configuration.yaml \
#     --input_dir=$ROOT_DIR/data/derived/project_portable_flume/blobs/ \
#     --input_model=$ROOT_DIR/models/mzb-pil/bm2ccwxc \
#     --output_dir=$ROOT_DIR/results/skeletons/project_portable_flume \
#     -v


