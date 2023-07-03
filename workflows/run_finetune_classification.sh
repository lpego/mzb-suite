#!/bin/bash
# set up pipeline to fine tune the classification model

ROOT_DIR="/data/shared/mzb-workflow"
MODEL="efficientnet-b2-v0"
LSET_FOLD=$ROOT_DIR/data/learning_sets/project_portable_flume/aggregated_learning_sets



## This has to be run once, to create the curated learning sets
## ------------------------------------------------------------

if [ -d "$LSET_FOLD" ];
then
    echo "Directory $LSET_FOLD exists." 
else 
    echo "Directory $LSET_FOLD is being set up."
    python $ROOT_DIR/scripts/classification/main_prepare_learning_sets.py \
    --input_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/curated_learning_sets/ \
    --taxonomy_file=$ROOT_DIR/data/MZB_taxonomy.csv \
    --output_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/aggregated_learning_sets \
    --config_file=$ROOT_DIR/configs/configuration_flume_datasets.yaml \
    # -v
fi

# This is run to fine tune the classification model. It will read the curated learning sets and will return a new model
# ---------------------------------------------------------------------------------------------------------------
python $ROOT_DIR/scripts/classification/main_classification_finetune.py \
    --config_file=$ROOT_DIR/configs/configuration_flume_datasets.yaml \
    --input_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/aggregated_learning_sets/ \
    --save_model=$ROOT_DIR/models/mzb-classification-models/$MODEL \
    # -v

# This is run to classify a custom folder structure and will regturn a csv with the results
# Specifically, this is run on the validation set to get the accuracy of the model
# ------------------------------------------------------------------------------------------
 python $ROOT_DIR/scripts/classification/main_classification_inference.py \
    --config_file=$ROOT_DIR/configs/configuration_flume_datasets.yaml \
    --input_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/aggregated_learning_sets/val_set \
    --input_model=$ROOT_DIR/models/mzb-classification-models/$MODEL \
    --output_dir=$ROOT_DIR/results/classification/project_portable_flume/ \
    -v

# And this is to classify the mixed set, a custom folder structure with images from different sources
# ---------------------------------------------------------------------------------------------------
python $ROOT_DIR/scripts/classification/main_classification_inference.py \
    --config_file=$ROOT_DIR/configs/configuration_flume_datasets.yaml \
    --input_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/aggregated_learning_sets/mixed_set \
    --input_model=$ROOT_DIR/models/mzb-classification-models/$MODEL \
    --output_dir=$ROOT_DIR/results/classification/project_portable_flume/ \
    -v


# --> the classifier using a ConvNext or EfficientNet-b2 model is best, so we will use that one probably. 