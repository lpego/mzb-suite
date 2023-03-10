# set up pipeline to fine tune the classification model

ROOT_DIR="/data/shared/mzb-workflow"
MODEL=efficientnet-b2-v0

## This has to be run once, to create the curated learning sets
## ------------------------------------------------------------
python scripts/classification/main_prepare_learning_sets.py \
    --input_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/curated_learning_sets_flume/ \
    --taxonomy_file=$ROOT_DIR/data/MZB_taxonomy.csv \
    --output_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/aggregated_learning_sets \
    --config_file=$ROOT_DIR/configs/global_configuration.yaml \
    -v

## This is run to fine tune the classification model. It will read the curated learning sets and will return a new model
## ---------------------------------------------------------------------------------------------------------------
python scripts/classification/main_classification_finetune.py \
    --config_file=$ROOT_DIR/configs/global_configuration.yaml \
    --input_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/aggregated_learning_sets/ \
    --save_model=$ROOT_DIR/models/mzb-class/$MODEL \
    -v

## This is run to classify a custom folder structure and will regturn a csv with the results
## Specifically, this is run on the validation set to get the accuracy of the model
## ------------------------------------------------------------------------------------------
 python scripts/classification/main_classification_inference.py \
    --config_file=$ROOT_DIR/configs/global_configuration.yaml \
    --input_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/aggregated_learning_sets/val_set \
    --input_model=$ROOT_DIR/models/mzb-class/$MODEL \
    --output_dir=$ROOT_DIR/results/classification/project_portable_flume/ \
    -v

## And this is to classify the mixed set, a custom folder structure with images from different sources
## ---------------------------------------------------------------------------------------------------
python scripts/classification/main_classification_inference.py \
    --config_file=$ROOT_DIR/configs/global_configuration.yaml \
    --input_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/aggregated_learning_sets/mixed_set \
    --input_model=$ROOT_DIR/models/mzb-class/$MODEL \
    --output_dir=$ROOT_DIR/results/classification/project_portable_flume/ \
    -v
