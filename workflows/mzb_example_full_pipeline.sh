#!/bin/bash 

# ## ------------------------------------------------------------------------------------ ##
## Definition of running parameters. 
## The path specified in ROOT_DIR is for virtual sessions on Renkulab, yours may differ! 
ROOT_DIR="/home/jovyan/work/mzb-workflow"
MODEL_C="convnext-small-v0" # classification model
MODEL_S="mit-b2-v0" # skeletonization model

# ## SEGMENTATION ##
# ## ------------------------------------------------------------------------------------ ##
## This extracts single organisms as clips from full-pane images; 
## to tweak the results, see the configuration file parameters. 
python ${ROOT_DIR}/scripts/image_parsing/main_raw_to_clips.py \
    --input_dir=${ROOT_DIR}/data/mzb_example_data/raw_img/ \
    --output_dir=${ROOT_DIR}/data/mzb_example_data/derived/blobs/ \
    --save_full_mask_dir=${ROOT_DIR}/data/mzb_example_data/derived/full_image_masks \
    --config_file=${ROOT_DIR}/configs/mzb_example_config.yaml \
    -v

# ## CLASSIFICATION ##
# ## ------------------------------------------------------------------------------------ ##
## This classifies organisms into taxonomic categories and returns a csv with the results; 
## if run on e.g. a validaton / test set, it will also produce accuracy metrics. 
## Make sure to pass this module only clips generated with main_raw_to_clips.py
python ${ROOT_DIR}/scripts/classification/main_classification_inference.py \
    --input_dir=${ROOT_DIR}/data/mzb_example_data/training_dataset/test_set/ \
    --input_model=${ROOT_DIR}/models/mzb-classification-models/${MODEL_C} \
    --taxonomy_file=${ROOT_DIR}/data/mzb_example_data/MZB_taxonomy.csv \
    --output_dir=${ROOT_DIR}/results/mzb_example/classification/test_set/ \
    --config_file=${ROOT_DIR}/configs/mzb_example_config.yaml \
    -v

# ## SKELETONIZATION ## 
# ## ------------------------------------------------------------------------------------ ##
## This runs the unsupervised skeletonization and measurement. It will read all the mask 
## clips created in the first step and will return a csv with the results. 
## This unsupervised pipeline can only approximate body length, not head width.
python ${ROOT_DIR}/scripts/skeletons/main_unsupervised_skeleton_estimation.py \
    --input_dir=${ROOT_DIR}/data/mzb_example_data/derived/blobs/ \
    --output_dir=${ROOT_DIR}/results/mzb_example/skeletons/unsupervised_skeletons/ \
    --save_masks=${ROOT_DIR}/data/mzb_example_data/derived/skeletons/unsupervised_skeletons/ \
    --config_file=${ROOT_DIR}/configs/mzb_example_config.yaml \
    --list_of_files=None \
    -v

# ## ------------------------------------------------------------------------------------ ##
## This runs a supervised DL model to predict body length and head width skeletons; 
## it stores the masks as images, and saves measurements into a csv file. 
python ${ROOT_DIR}/scripts/skeletons/main_supervised_skeleton_inference.py \
    --input_dir=${ROOT_DIR}/data/mzb_example_data/derived/blobs/ \
    --input_type="external" \
    --input_model=${ROOT_DIR}/models/mzb-skeleton-models/${MODEL_S} \
    --output_dir=${ROOT_DIR}/results/mzb_example/skeletons/supervised_skeletons/ \
    --save_masks=${ROOT_DIR}/data/mzb_example_data/derived/skeletons/supervised_skeletons/ \
    --config_file=${ROOT_DIR}/configs/mzb_example_config.yaml \
    -v
