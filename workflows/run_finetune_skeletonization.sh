# set up pipeline to fine tune the supervised skeletonization model

ROOT_DIR="/data/shared/mzb-workflow"
MODEL=mit-b2-v0

## This has to be run once, to create the curated learning sets
## ------------------------------------------------------------
python scripts/skeletons/main_preprocess_manual_skeleton_annotations.py \
    --input_raw_dir=$ROOT_DIR/data/raw/2021_swiss_invertebrates/manual_measurements/ \
    --input_clips_dir=$ROOT_DIR/data/derived/project_portable_flume/blobs/ \
    --output_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/skeletonization/ \
    --config_file=$ROOT_DIR/configs/global_configuration.yaml \
    -v

## This is run to fine tune the classification model. It will read the curated learning sets and will return a new model
## ---------------------------------------------------------------------------------------------------------------
python scripts/skeletons/main_supervised_skeletons_finetune.py \
    --config_file=$ROOT_DIR/configs/global_configuration.yaml \
    --input_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/skeletonization/ \
    --save_model=$ROOT_DIR/models/mzb-skels/$MODEL \
    -v

## This is run on a custom folder structure and will regturn a csv with the results
## Specifically, this is run on the validation set to get the accuracy of the model
## ------------------------------------------------------------------------------------------
python scripts/skeletons/main_supervised_skeleton_inference.py \
    --config_file=$ROOT_DIR/configs/global_configuration.yaml \
    --input_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/skeletonization/ \
    --input_model=$ROOT_DIR/models/mzb-skels/$MODEL \
    --output_dir=$ROOT_DIR/results/skeletons/project_portable_flume/supervised_skeletons/ \
    -v


# ## And this is to parse the mixed set, a custom folder structure with images from different sources
# ## ---------------------------------------------------------------------------------------------------
# python scripts/skeletons/main_supervised_skeleton_inference.py \
#     --config_file=$ROOT_DIR/configs/global_configuration.yaml \
#     --input_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/skeletonization/ \
#     --input_model=$ROOT_DIR/models/mzb-skels/$MODEL \
#     --output_dir=$ROOT_DIR/results/skeletons/project_portable_flume/supervised_skeletons/ \
#     -v
