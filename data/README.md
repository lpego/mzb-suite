# Data folder for mzb-suite

The folder `mzb_example_data` contains images that can be used to run the tutorials (i.e. notebooks) and test functionality. 
Depending on configuration, it may be required to pull them using `git lfs` if they are not downloaded together with the rest of the repo. 

 - `mzb_example_data/raw_img` contains three full sized images with several organisms each. This is the kind of images the pipeline is designed to handle, from segmentation to segmentation, skeletoniztion and classificaiton 
 - `mzb_example_data/training_dataset` contains the `trn_set` and a `val_set` directories, each containing the same number of folders which are named after the class (i.e. taxon in this case) of the images they contain; this dataset is to be used to test the re-training of the models with own data. 
 