#!/bin/bash 

ROOT_DIR="/data/shared/mzb-workflow"
MODEL="efficientnet-b2-v0"
# LSET_FOLD=${ROOT_DIR}/data/mzb_example_data/training_dataset

echo

# ## SEGMENTATION ##
# ## ------------------------------------------------------------------------------------ ##

# python scripts/image_parsing/main_raw_to_clips.py \
#     --input_dir=${ROOT_DIR}/data/mzb_example_data/raw_img \
#     --output_dir=${ROOT_DIR}/data/derived/mzb_example_data/blobs/ \
#     --save_full_mask_dir=${ROOT_DIR}/data/derived/mzb_example_data/full_image_masks/ \
#     --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml #\ 
#     # -v

# ## CLASSIFICATION ##
# ## ------------------------------------------------------------------------------------ ##

### This is run to classify a custom folder structure and will return a csv with the results 
# ## (filename, predicted class, probability of prediction)
# ## eg. on a validaton / test set, or on a new set of images, or whatever. be assured to pass only clips generated similarly 
# ## to the first step (main_raw_to_clips.py)
# ## classification models are stored in ${ROOT_DIR}/models/mzb-classification/
# python scripts/classification/main_classification_inference.py \
#     --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml \
#     --input_dir=${ROOT_DIR}/data/mzb_example_data/training_dataset/val_set/ \
#     --input_model=${ROOT_DIR}/models/mzb-classification-models/${MODEL} \
#     --output_dir=${ROOT_DIR}/results/classification/mzb_example_data/ #\
#     # -v

# ## SKELETONIZATION ## 
# ## ------------------------------------------------------------------------------------ ##

# ## This part run the unsupervised skeletonization and measurement. It will read all the mask clips created at the first step
# ## and will return a csv with the results (filename, skeleton, etc)
# ## The pipeline to get these numbers is unsupervised, but can only approximate length of the organism, and not the width of the head.
# ## For this, the next step that uses a supervised neural network is required.
# ## this function also takes a list of files to process, if you want to run it on a subset of the data, passed as csv with column "file"
# python scripts/skeletons/main_unsupervised_skeleton_estimation.py \
#     --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml \
#     --input_dir=${ROOT_DIR}/data/derived/mzb_example_data/blobs/ \
#     --output_dir=${ROOT_DIR}/results/mzb_example_data/skeletons/unsupervised_skeletons/ \
#     --save_masks=${ROOT_DIR}/data/derived/mzb_example_data/skeletons/unsupervised_skeletons/ #\
#     --list_of_files=None \
#     # -v

# ## This is run on a custom folder structure and will return a csv with the results
# ## Specifically, this is run on the validation set to get the accuracy of the model
# python scripts/skeletons/main_supervised_skeleton_inference.py \
#     --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml \
#     --input_dir=${ROOT_DIR}/data/mzb_example_data/training_dataset/trn_set/diptera/ \
#     --input_type="val" \
#     --input_model=${ROOT_DIR}/models/mzb-skeleton-models/mit-b2-v1/ \
#     --output_dir=${ROOT_DIR}/results/mzb_example_data/skeletons/supervised_skeletons/skseg_trn_set \
#     --save_masks=${ROOT_DIR}/data/derived/mzb_example_data/skeletons/supervised_skeletons/trn_set_masks/ #\ 
#     # -v

## And this is to parse a custom folder structure with images from different sources
# python scripts/skeletons/main_supervised_skeleton_inference.py \
#     --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml \
#     --input_dir=${ROOT_DIR}/data/mzb_example_data/training_dataset/trn_set/diptera/ \
#     --input_type="external" \
#     --input_model=${ROOT_DIR}/models/mzb-skeleton-models/mit-b2-v1 \
#     --output_dir=${ROOT_DIR}/results/mzb_example_data/skeletons/supervised_skeletons/skseg_trn_set \
#     --save_masks=${ROOT_DIR}/data/derived/mzb_example_data/skeletons/supervised_skeletons/trn_set_masks/ #\ 
#     # -v
