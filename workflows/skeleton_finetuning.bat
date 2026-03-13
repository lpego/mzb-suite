@echo off

@REM ## ------------------------------------------------------------------------------------ ##
@REM ## Definition of running parameters. 
@REM ## The LSET_FOLD should contain the train, test and val folders
SET ROOT_DIR=D:\mzb-suite
SET LSET_FOLD=%ROOT_DIR%\data\mzb_example_data\skels_training_dataset\
SET MODEL_S=mit-b2-v0

@REM ## SKELETONIZATION FINETUNE ##
@REM ## ------------------------------------------------------------------------------------ ##
@REM ## This script uses the curated learning sets prepared to fine-tune a new model. 
python %ROOT_DIR%\scripts\skeletonization\main_supervised_skeletons_finetune.py^
 --config_file %ROOT_DIR%\configs\mzb_example_config.yaml^
 --input_dir %LSET_FOLD%^
 --save_model %ROOT_DIR%\models\mzb-skeleton-models\%MODEL_S%^
  -v