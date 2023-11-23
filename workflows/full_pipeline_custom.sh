#!/bin/bash 

# Definition of recurrent patterns

ROOT_DIR="/work/mzb-workflow"
MODEL_C="convnext-small-v0"
MODEL_S="mit-b2-v0"
## ------------------------------------------------------

python scripts/image_parsing/main_raw_to_clips.py \
    --input_dir=${ROOT_DIR}/data/mzb_example_data/raw_img/ \
    --output_dir=${ROOT_DIR}/data/mzb_example_data/derived/blobs/ \
    --save_full_mask_dir=${ROOT_DIR}/data/mzb_example_data/derived/full_image_masks_test \
    --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml #\
    # -v


## This is run to classify a custom folder structure and will regturn a csv with the results 
## (filename, predicted class, probability of prediction)
## eg. on a validaton / test set, or on a new set of images, or whatever. be assured to pass only clips generated similarty 
## to the first step (main_raw_to_clips.py)
## classification models are stored in ${ROOT_DIR}/models/mzb-class/
## ------------------------------------------------------------------------------------------
python scripts/classification/main_classification_inference.py \
    --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml \
    --input_dir=${ROOT_DIR}/data/mzb_example_data/training_dataset/val_set/ \
    --input_model=${ROOT_DIR}/models/mzb-classification-models/$MODEL_C \
    --output_dir=${ROOT_DIR}/results/mzb_example/classification/val_set/ \
    -v

## This part run the unsupervised skeletonization and measurement. It will read all the mask clips created at the first step
## and will return a csv with the results (filename, skeleton, etc)
## The pipeline to get these numbers is unsupervised, but can only approximate length of the insect, and not the width of the head.
## For this, the next step that uses a supervised neural network is required.
## this function also takes a list of files to process, if you want to run it on a subset of the data. As csv with column "file"
## ------------------------------------------------------------------------------------------
python scripts/skeletons/main_unsupervised_skeleton_estimation.py \
    --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml \
    --input_dir=${ROOT_DIR}/data/mzb_example_data/derived/blobs/ \
    --output_dir=${ROOT_DIR}/results/mzb_example/skeletons/automatic_skeletons/ \
    --save_masks=${ROOT_DIR}/data/mzb_example_data/derived/skeletons/automatic_skeletons/ \
    --list_of_files=None \
    -v


## And this is to use a supervised model to parse skeletons. It predicts both body length and head width
## and stores the masks as images, and saves measurements into a csv
## This is to parse a folder containing only images (not subdirectories)
## ---------------------------------------------------------------------------------------------------
python scripts/skeletons/main_supervised_skeleton_inference.py \
    --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml \
    --input_dir=${ROOT_DIR}/data/mzb_example_data/training_dataset/val_set/plecoptera/ \
    --input_type="external" \
    --input_model=${ROOT_DIR}/models/mzb-skeleton-models/$MODEL_S \
    --output_dir=${ROOT_DIR}/results/mzb_example/skeletons/supervised_skeletons/skseg_$MODEL_S_val_set\
    --save_masks=${ROOT_DIR}/data/mzb_example_data/derived/skeletons/supervised_skeletons/$MODEL_S/val_set_masks/ \
