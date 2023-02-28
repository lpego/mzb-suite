#!/bin/bash 

ROOT_DIR="/data/shared/mzb-workflow"

# python scripts/image_parsing/main_raw_to_clips.py \
#     --input_dir=$ROOT_DIR/data/raw/project_portable_flume \
#     --output_dir=$ROOT_DIR/data/blobs/project_portable_flume \
#     --config_file=$ROOT_DIR/configs/global_configuration.yml \
#     -v


python scripts/classification/main_prepare_learning_sets.py \
    --input_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/curated_learning_sets_flume/ \
    --taxonomy_file=$ROOT_DIR/data/MZB_taxonomy.csv \
    --output_dir=$ROOT_DIR/data/learning_sets/project_portable_flume/aggregated_learning_sets \
    --config_file=$ROOT_DIR/configs/global_configuration.yml \
    -v
