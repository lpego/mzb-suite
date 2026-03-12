@echo off

@REM ## ------------------------------------------------------------------------------------ ##
@REM ## Definition of running parameters. 
@REM ## You should substitute ROOT_DIR with the absolute path to your working directory 
SET ROOT_DIR=your_path_here
SET MODEL_C=convnext-small-v0
SET MODEL_S=mit-b2-v0

@REM ## SEGMENTATION ##
@REM ## ------------------------------------------------------------------------------------ ##
@REM ## This extracts single organisms as clips from full-pane images; 
@REM ## to tweak the results, see the configuration file parameters. 
python %ROOT_DIR%\scripts\segmentation\main_raw_to_clips.py^
 --input_dir %ROOT_DIR%\data\mzb_example_data\raw_img\^
 --output_dir %ROOT_DIR%\data\mzb_example_data\derived\blobs\^
 --save_full_mask_dir %ROOT_DIR%\data\mzb_example_data\derived\full_image_masks^
 --config_file %ROOT_DIR%\configs\mzb_example_config.yaml^
 -v

@REM ## CLASSIFICATION ##
@REM ## ------------------------------------------------------------------------------------ ##
@REM ## This classifies organisms into taxonomic categories and returns a csv with the results; 
@REM ## if run on e.g. a validaton \ test set, it will also produce accuracy metrics. 
@REM ## Make sure to pass this module only clips generated with main_raw_to_clips.py
python %ROOT_DIR%\scripts\classification\main_classification_inference.py^
 --input_dir %ROOT_DIR%\data\mzb_example_data\derived\blobs\^
 --input_model %ROOT_DIR%\models\mzb-classification-models\%MODEL_C%^
 --taxonomy_file %ROOT_DIR%\data\mzb_example_data\MZB_taxonomy.csv^
 --output_dir %ROOT_DIR%\results\mzb_example\classification\^
 --config_file %ROOT_DIR%\configs\mzb_example_config.yaml^
 -v

@REM ## SKELETONIZATION UNSUPERVISED ## 
@REM ## ------------------------------------------------------------------------------------ ##
@REM ## This runs the unsupervised skeletonization and measurement. It will read all the mask 
@REM ## clips created in the first step and will return a csv with the results. 
@REM ## This unsupervised pipeline can only approximate body length, not head width.
python %ROOT_DIR%\scripts\skeletonization\main_unsupervised_skeleton_estimation.py^
 --input_dir %ROOT_DIR%\data\mzb_example_data\derived\blobs\^
 --output_dir %ROOT_DIR%\results\mzb_example\skeletons\unsupervised_skeletons\^
 --save_masks %ROOT_DIR%\data\mzb_example_data\derived\skeletons\unsupervised_skeletons\^
 --config_file %ROOT_DIR%\configs\mzb_example_config.yaml^
 --list_of_files None^
 -v

@REM ## SKELETONIZATION SUPERVISED ## 
@REM ## ------------------------------------------------------------------------------------ ##
@REM ## This runs a supervised DL model to predict body length and head width skeletons; 
@REM ## it stores the masks as images, and saves measurements into a csv file. 
python %ROOT_DIR%\scripts\skeletonization\main_supervised_skeleton_inference.py^
 --input_dir %ROOT_DIR%\data\mzb_example_data\derived\blobs\^
 --input_type external^
 --input_model %ROOT_DIR%\models\mzb-skeleton-models\%MODEL_S%^
 --output_dir %ROOT_DIR%\results\mzb_example\skeletons\supervised_skeletons\^
 --save_masks %ROOT_DIR%\data\mzb_example_data\derived\skeletons\supervised_skeletons\^
 --config_file %ROOT_DIR%\configs\mzb_example_config.yaml^
 -v

@REM ## SUMMARISATION ##
@REM ## ------------------------------------------------------------------------------------ ##
@REM ## This module collects the outputs from the others and merges them into a csv file. 
python %ROOT_DIR%\scripts\summarisation.py^
 --classification %ROOT_DIR%\results\mzb_example\classification^
 --skeletons_supervised %ROOT_DIR%\results\mzb_example\skeletons\supervised_skeletons^
 --skeletons_unsupervised %ROOT_DIR%\results\mzb_example\skeletons\unsupervised_skeletons^
 --output_folder %ROOT_DIR%\results\mzb_example^
 --config_file %ROOT_DIR%\configs\mzb_example_config.yaml^
 -v