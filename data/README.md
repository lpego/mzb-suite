# Data folder for mzb-suite

The folder `mzb_example_data` contains images that can be used to run the tutorials (i.e. notebooks) and test functionality. 
Depending on local git configuration, it may be required to pull them using `git lfs` if they are not downloaded together with the rest of the repo. 

<<<<<<< HEAD
 - `mzb_example_data/raw_img` contains three full sized images with several organisms each. This is the kind of images the pipeline is designed to handle, from segmentation to skeletoniztion and classification. 
 - `mzb_example_data/training_dataset` contains the `trn_set` and a `val_set` directories, each containing the same number of folders which are named after the class (i.e. taxon in this case) of the images they contain; this dataset is to be used to test the re-training of the models with own data. 
=======
 - `mzb_example_data/raw_img` contains three full sized images with several organisms each. This is the kind of images the pipeline is designed to handle, from segmentation to skeletonization and classification 
 - `mzb_example_data/training_dataset` contains the `trn_set` and a `val_set` directories, each containing the same exact folders which are named after the class (i.e. taxon, in this case order or higher) of the images they contain; this dataset is to be used to test the re-training of the models with own data. 
>>>>>>> 7f578398db47be6ed7c479b3a4a4da3b255875f4
 