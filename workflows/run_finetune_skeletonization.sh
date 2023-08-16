#!/bin/bash

## Set up pipeline to fine tune the supervised skeletonization model
ROOT_DIR="/data/shared/mzb-workflow"
MODEL=mit-b2-v1-test
LSET_FOLD=${ROOT_DIR}/data/learning_sets/project_portable_flume/skeletonization

## This has to be run once, to create the curated learning sets, 
## ------------------------------------------------------------
if [ -d ${LSET_FOLD} ];
then
    echo "Directory ${LSET_FOLD} exists." 
else 
    echo "Directory ${LSET_FOLD} is being set up."
    python scripts/skeletons/main_preprocess_manual_skeleton_annotations.py \
        --input_raw_dir=${ROOT_DIR}/data/raw/2021_swiss_invertebrates/manual_measurements/ \
        --input_clips_dir=${ROOT_DIR}/data/derived/project_portable_flume/blobs/ \
        --output_dir=${LSET_FOLD} \
        --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml \
        # -v
fi

## This is run to fine tune the skeleton prediction model. It will read the curated learning sets and will return a new model
## ---------------------------------------------------------------------------------------------------------------
python scripts/skeletons/main_supervised_skeletons_finetune.py \
    --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml \
    --input_dir=${LSET_FOLD} \
    --save_model=${ROOT_DIR}/models/mzb-skeleton-models/${MODEL} \
    # -v

## This is run on a custom folder structure and will regturn a csv with the results
## Specifically, this is run on the validation set to get the accuracy of the model
## ------------------------------------------------------------------------------------------
# python scripts/skeletons/main_supervised_skeleton_inference.py \
#     --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml \
#     --input_dir=${LSET_FOLD} \
#     --input_type="val" \
#     --input_model=${ROOT_DIR}/models/mzb-skeleton-models/${MODEL} \
#     --output_dir=${ROOT_DIR}/results/project_portable_flume/skeletons/supervised_skeletons/skseg_${MODEL}_validation_set/ \
#     --save_masks=${ROOT_DIR}/data/derived/project_portable_flume/skeletons/supervised_skeletons/skseg_${MODEL}/validation_set \
#     -v

# ## And this is to parse a custom folder structure with images from different sources. Turn True to run it. Takes some time.
# ## ---------------------------------------------------------------------------------------------------
if [ false ] ; # change to true to run inference on external set
then
    echo "Skipping inference on external set"
else
    python scripts/skeletons/main_supervised_skeleton_inference.py \
        --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml \
        --input_dir=${ROOT_DIR}/data/learning_sets/project_portable_flume/aggregated_learning_sets/mixed_set/ \
        --input_type="external" \
        --input_model=${ROOT_DIR}/models/mzb-skeleton-models/${MODEL} \
        --output_dir=${ROOT_DIR}/results/project_portable_flume/skeletons/supervised_skeletons/skseg_${MODEL}_external_mixed \
        --save_masks=${ROOT_DIR}/data/derived/project_portable_flume/skeletons/supervised_skeletons/skseg_${MODEL}/mixed_set_masks/ \
        # -v
fi

## Validate the model on the validation set 
## ----------------------------------------
# python scripts/skeletons/main_supervised_skeleton_assessment.py \
#     --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml \
#     --model_annotations=${ROOT_DIR}/results/project_portable_flume/skeletons/supervised_skeletons/skseg_${MODEL}_validation_set/size_skel_supervised_model.csv \
#     --manual_annotations=${ROOT_DIR}/data/learning_sets/project_portable_flume/skeletonization/manual_annotations_summary.csv