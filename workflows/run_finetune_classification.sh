#!/bin/bash

# ## ------------------------------------------------------------------------------------ ##
## This workflow sets up and runs the finetuning of the classification model
## The path specified in ROOT_DIR is for virtual sessions on Renkulab, yours  may differ! 
# ROOT_DIR="/home/jovyan/work/mzb-workflow"
ROOT_DIR="/data/shared/mzb-workflow/"
MODEL="convnext_small_v0_prerelease"
# LSET_FOLD="${ROOT_DIR}/data/mzb_example_data/aggregated_learning_set"
LSET_FOLD="${ROOT_DIR}/data/learning_sets/project_portable_flume/aggregated_learning_sets"

# ## ------------------------------------------------------------------------------------ ##
## This runs only if the target aggregate learning sets folder doesn't exist yet, 
## and then creates it based on the specified taxonomic rank and taxonomy file. 
if [ -d "${LSET_FOLD}" ];
then
    echo "Directory ${LSET_FOLD} exists." 
else 
    echo "Directory ${LSET_FOLD} is being set up."
    ## this creates the aggregated training set
    python ${ROOT_DIR}/scripts/classification/main_prepare_learning_sets.py \
    --input_dir=${ROOT_DIR}/data/mzb_example_data/training_dataset/trn_set/ \
    --taxonomy_file=${ROOT_DIR}/data/mzb_example_data/MZB_taxonomy.csv \
    --output_dir=${ROOT_DIR}/data/mzb_example_data/aggregated_set/trn_set/ \
    --config_file=${ROOT_DIR}/configs/mzb_example_config.yaml \
    # -v
    ## this created the aggreagted validation set
    python ${ROOT_DIR}/scripts/classification/main_prepare_learning_sets.py \
    --input_dir=${ROOT_DIR}/data/mzb_example_data/training_dataset/val_set/ \
    --taxonomy_file=${ROOT_DIR}/data/mzb_example_data/MZB_taxonomy.csv \
    --output_dir=${ROOT_DIR}/data/mzb_example_data/aggregated_set/val_set/ \
    --config_file=${ROOT_DIR}/configs/mzb_example_config.yaml \
    # -v
fi

# ## ------------------------------------------------------------------------------------ ##
## This finetunes the model based on the aggreagted learning sets; it returns a new model.
python ${ROOT_DIR}/scripts/classification/main_classification_finetune.py \
    --input_dir=${LSET_FOLD} \
    --save_model=${ROOT_DIR}/models/mzb-classification-models/${MODEL} \
    --config_file=${ROOT_DIR}/configs/mzb_example_config.yaml \

# ## ------------------------------------------------------------------------------------ ##
## This uses the newly trained model to classify new images; 
## specifically, this runs on the validation set to get the accuracy of the model. 
python ${ROOT_DIR}/scripts/classification/main_classification_inference.py \
    --input_dir=${LSET_FOLD}/val_set \
    --input_model=${ROOT_DIR}/models/mzb-classification-models/${MODEL} \
    --output_dir=${ROOT_DIR}/results/project_portable_flume/classification/${MODEL}_temp/ \
    --config_file=${ROOT_DIR}/configs/mzb_example_config.yaml \
    --taxonomy_file=${ROOT_DIR}/data/mzb_example_data/MZB_taxonomy.csv \

# # ## ------------------------------------------------------------------------------------ ##
# ## And this can be adapted to your own external set of images, 
# ## a custom folder structure with images from different sources.
# python ${ROOT_DIR}/scripts/classification/main_classification_inference.py \
#     --input_dir=${ROOT_DIR}/data/YOUR_IMAGE_SET_HERE/ \
#     --taxonomy_file=${ROOT_DIR}/data/mzb_example_data/YOUR_TAXONOMY_FILE_HERE.csv \
#     --input_model=${ROOT_DIR}/models/mzb-classification-models/${MODEL}_aggregated \
#     --output_dir=${ROOT_DIR}/results/mzb_example_data/classification/${MODEL}_aggregated_external/ \
#     --config_file=${ROOT_DIR}/configs/mzb_example_config.yaml \
#     # -v
