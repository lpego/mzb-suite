@echo off

@REM ## ------------------------------------------------------------------------------------ ##
@REM ## Definition of running parameters. 
@REM ## The path specified in ROOT_DIR is for virtual sessions on Renkulab, yours may differ! 
SET ROOT_DIR=D:\mzb-workflow
SET MODEL_C=convnext-small-v0
SET MODEL_S=mit-b2-v0

@REM ## ------------------------------------------------------------------------------------ ##
@REM ## Preprocess manual annotations for supervised skeletons assessment;  
python %ROOT_DIR%\scripts\skeletons\main_preprocess_manual_skeleton_annotations.py^
 --input_raw_dir D:\phenopype\line_annotations\^
 --input_clips_dir D:\phenopype\data\^
 --skel_save_attributes %ROOT_DIR%\data\mzb_example\skeletons\supervised_skeletons\manual_anns^
 --output_dir %ROOT_DIR%\data\mzb_example\skeletons\supervised_skeletons\manual_anns^
 --config_file %ROOT_DIR%\configs\mzb_example_config.yaml^
 -v