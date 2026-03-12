@echo off

@REM ## ------------------------------------------------------------------------------------ ##
@REM ## Definition of running parameters. 
@REM ## The path specified in ROOT_DIR is for virtual sessions on Renkulab, yours may differ! 
SET ROOT_DIR=D:\mzb-suite

@REM ## ------------------------------------------------------------------------------------ ##
@REM ## Preprocess manual annotations for supervised skeletons assessment;  
python %ROOT_DIR%\scripts\skeletonization\main_preprocess_manual_skeleton_annotations.py^
 --input_raw_dir D:\phenopype\line_annotations\^
 --input_clips_dir D:\phenopype\data\^
 --skel_save_attributes %ROOT_DIR%\results\mzb_example\skeletons\supervised_skeletons\assessment^
 --output_dir %ROOT_DIR%\results\mzb_example\skeletons\supervised_skeletons\assessment^
 --config_file %ROOT_DIR%\configs\mzb_example_config.yaml^
 -v

@REM @REM ## ------------------------------------------------------------------------------------ ##
@REM @REM ## Assess supervised skeletons accuracy  
@REM python %ROOT_DIR%\scripts\skeletonization\main_supervised_skeleton_assessment.py^
@REM  --model_annotations %ROOT_DIR%\results\mzb_example\skeletons\supervised_skeletons\^
@REM  --manual_annotations %ROOT_DIR%\data\mzb_example_data\derived\skeletons\supervised_skeletons\manual_anns^
@REM  --output_dir %ROOT_DIR%\results\mzb_example_data\skeletons\supervised_skeletons\^
@REM  --config_file %ROOT_DIR%\configs\mzb_example_config.yaml^
@REM  -v