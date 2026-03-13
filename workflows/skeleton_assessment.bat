@echo off

@REM ## ------------------------------------------------------------------------------------ ##
@REM ## Definition of running parameters. 
@REM ## The path specified in ROOT_DIR will differ from your local path.
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

@REM ## ------------------------------------------------------------------------------------ ##
@REM ## Assess supervised skeletons accuracy  
python %ROOT_DIR%\scripts\skeletonization\main_supervised_skeleton_assessment.py^
 --model_annotations %ROOT_DIR%\results\mzb_example\skeletons\supervised_skeletons\blobs_supervised_20260312_1853\supervised_skeletons.csv^
 --manual_annotations %ROOT_DIR%\results\mzb_example\skeletons\supervised_skeletons\assessment\manual_annotations_summary.csv^
 --output_dir %ROOT_DIR%\results\mzb_example\skeletons\supervised_skeletons\assessment^
 --config_file %ROOT_DIR%\configs\mzb_example_config.yaml^
 -v