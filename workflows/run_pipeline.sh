#!/bin/bash 

ROOT_DIR="/data/shared/mzb-workflow"
# MODEL="efficientnet-b2-v0"
# LSET_FOLD=$ROOT_DIR/data/learning_sets/project_portable_flume/aggregated_learning_sets

# python scripts/image_parsing/main_raw_to_clips.py \
#     --input_dir=$ROOT_DIR/data/raw/project_portable_flume \
#     --output_dir=$ROOT_DIR/data/derived_v2/project_portable_flume/blobs/ \
#     --save_full_mask_dir=$ROOT_DIR/data/derived_v2/project_portable_flume/full_image_masks \
#     --config_file=$ROOT_DIR/configs/configuration_flume_datasets.yaml \
#     -v

## CHECK: _mask_properties.csv file in the output directory, does not get filled correctly  

# This is run to classify a custom folder structure and will regturn a csv with the results 
# (filename, predicted class, probability of prediction)
# eg. on a validaton / test set, or on a new set of images, or whatever. be assured to pass only clips generated similarty 
# to the first step (main_raw_to_clips.py)
# classification models are stored in $ROOT_DIR/models/mzb-class/
# python scripts/classification/main_classification_inference.py \
#     --config_file=$ROOT_DIR/configs/configuration_flume_datasets.yaml \
#     --input_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/aggregated_learning_sets/val_set \
#     --input_model=$ROOT_DIR/models/mzb-class/convnext-small \
#     --output_dir=$ROOT_DIR/results/classification/project_portable_flume/ \
#     -v

### Ex. run on all the clips available to get full predictions
## python scripts/classification/main_classification_inference.py \
##     --config_file=$ROOT_DIR/configs/configuration_flume_datasets.yaml \
##     --input_dir=$ROOT_DIR/data/derived/project_portable_flume/blobs/ \
##     --input_model=$ROOT_DIR/models/mzb-class/bm2ccwxc \
##     --output_dir=$ROOT_DIR/results/classification/project_portable_flume/ \
##     -v

# This part run the unsupervised skeletonization and measurement. It will read all the mask clips created at the first step
# and will return a csv with the results (filename, skeleton, etc)
# The pipeline to get these numbers is unsupervised, but can only approximate length of the insect, and not the width of the head.
# For this, the next step that uses a supervised neural network is required.
# this function also takes a list of files to process, if you want to run it on a subset of the data. As csv with column "file"
## ------------------------------------------------------------------------------------------
python scripts/skeletons/main_unsupervised_skeleton_estimation.py \
    --config_file=$ROOT_DIR/configs/configuration_flume_datasets.yaml \
    --input_dir=$ROOT_DIR/data/derived/project_portable_flume/blobs/ \
    --output_dir=$ROOT_DIR/results/project_portable_flume/skeletons/automatic_skeletons/ \
    --save_masks=$ROOT_DIR/data/derived_v2/project_portable_flume/skeletons/automatic_skeletons/ \
    --list_of_files=None \
    # -v

## This is run on a custom folder structure and will regturn a csv with the results
## Specifically, this is run on the validation set to get the accuracy of the model
## ------------------------------------------------------------------------------------------
# python scripts/skeletons/main_supervised_skeleton_inference.py \
#     --config_file=$ROOT_DIR/configs/configuration_flume_datasets.yaml \
#     --input_dir=$LSET_FOLD \
#     --input_type="val" \
#     --input_model=$ROOT_DIR/models/mzb-skeleton-models/$MODEL \
#     --output_dir=$ROOT_DIR/data/derived/skeletons/project_portable_flume/supervised_skeletons/test_skseg_$MODEL \
#     --save_masks=$ROOT_DIR/data/derived_v2/project_portable_flume/skeletons/supervised_skeletons/skseg_efficientnet-b2-v2/mixed_set_masks/ \

# #     -v

# ## And this is to parse a custom folder structure with images from different sources
# ## ---------------------------------------------------------------------------------------------------
# python scripts/skeletons/main_supervised_skeleton_inference.py \
#     --config_file=$ROOT_DIR/configs/configuration_flume_datasets.yaml \
#     --input_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/aggregated_learning_sets/mixed_set/ \
#     --input_type="external" \
#     --input_model=$ROOT_DIR/models/mzb-skeleton-models/$MODEL \
#     --output_dir=$ROOT_DIR/results/project_portable_flume/skeletons/supervised_skeletons/skseg_efficientnet-b2-v2/ \
#     --save_masks=$ROOT_DIR/data/derived_v2/project_portable_flume/skeletons/supervised_skeletons/skseg_efficientnet-b2-v2/mixed_set_masks/ \
    # -v