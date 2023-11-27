#!/bin/bash 

## Definition of running parameters. 
## The root path specified here in ROOT_DIR is for working with notebooks on virtual sessions on Renkulab, your path may differ! 
ROOT_DIR="/home/jovyan/work/mzb-workflow"
MODEL_C="convnext-small-v0"
MODEL_S="mit-b2-v0"

## ------------------------------------------------------
python scripts/image_parsing/main_raw_to_clips.py \
    --input_dir=${ROOT_DIR}/data/mzb_example_data/raw_img/ \
    --output_dir=${ROOT_DIR}/data/mzb_example_data/derived/blobs/ \
    --save_full_mask_dir=${ROOT_DIR}/data/mzb_example_data/derived/full_image_masks \
    --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml \
    -v

## This is run to classify organisms into taxonimic categories and will return a csv with the results (filename, predicted class, probability of prediction) 
## if run on eg. on a validaton / test set, it will also produce accuracy metrics. 
## Make sure to pass this module only similarly generated clips as in the the first step (main_raw_to_clips.py); 
## classification models are stored in ${ROOT_DIR}/models/mzb-class/
## ------------------------------------------------------------------------------------------
python scripts/classification/main_classification_inference.py \
    --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml \
    --input_dir=${ROOT_DIR}/data/mzb_example_data/training_dataset/val_set/ \
    --input_model=${ROOT_DIR}/models/mzb-classification-models/$MODEL_C \
    --output_dir=${ROOT_DIR}/results/mzb_example/classification/val_set/ \
    -v

## This part runs the unsupervised skeletonization and measurement. It will read all the mask clips created in the first step
## and will return a csv with the results (filename, skeleton, etc). 
## This unsupervised pipeline can only approximate length of the insect, and not the width of the head.
## ------------------------------------------------------------------------------------------
python scripts/skeletons/main_unsupervised_skeleton_estimation.py \
    --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml \
    --input_dir=${ROOT_DIR}/data/mzb_example_data/derived/blobs/ \
    --output_dir=${ROOT_DIR}/results/mzb_example/skeletons/automatic_skeletons/ \
    --save_masks=${ROOT_DIR}/data/mzb_example_data/derived/skeletons/automatic_skeletons/ \
    --list_of_files=None \
    -v

## This runs a supervised DL model to predict body length and head width skeletons; 
## it stores the masks as images, and saves measurements into a csv file. 
## ---------------------------------------------------------------------------------------------------
python scripts/skeletons/main_supervised_skeleton_inference.py \
    --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml \
    --input_dir=${ROOT_DIR}/data/mzb_example_data/training_dataset/val_set/plecoptera/ \
    --input_type="external" \
    --input_model=${ROOT_DIR}/models/mzb-skeleton-models/$MODEL_S \
    --output_dir=${ROOT_DIR}/results/mzb_example/skeletons/supervised_skeletons/skseg_$MODEL_S_val_set\
    --save_masks=${ROOT_DIR}/data/mzb_example_data/derived/skeletons/supervised_skeletons/$MODEL_S/val_set_masks/ \
