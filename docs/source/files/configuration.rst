Configuration
#############

All information related to a project is contained in a configuration file, located in ``/configs/{configuration_file.yaml}``. This file, together with input/output directories and other parameters specified directly via CLI (i.e. Command Line Interface) or via shell script (see also :doc:`Workflows and Models <workflow_models>`), passes the necessary parameters to the scripts. 

We provide a template configuration file as well as a complete configuration file for the example project, *Portable Flume*. 

Parameters explanation
**********************

This list is structured as follows: 

 .. code-block:: yaml
    
        parameter_name: [admissible_value_1, admissible_value_2] 

Description of parameter, suggested values and rationale. 

.. admonition:: \ \ 

    Need to figure out a better way to format this than using code blocks and go through all the pars again... 

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This first block contains some general parameters:
.. # Arguments not to be spec via CLI

 - ``glob_random_seed``: this is just a arbitrary number used by model trainers, important for reproducibility. 
 - ``glob_root_folder``: this is the root folder of the project, it could be for example ``/home/user/my_project/``. 
 - ``glob_blobs_folder``: this is the location where you want the clips of the segmented organisms to be saved; we strongly recommend putting this inside of the main data folder, for example ``/data/shared/mzb-workflow/data/derived/blobs/``. 
 - ``glob_local_format``: what format do you want the plotting outputs to be saved in; acceptable values are: ``pdf``, ``jpg``, ``png`` and other common formats. **NEED TO DOUBLE CHECK THIS**
 - ``model_logger``: which data logger is used to track model training progress; for the moment, only ``wandb`` (`Weights & Biases <https://wandb.ai/site>`_) is supported. Note that W&B requires an account and to be setup by the user, see **WEIGHTS_&_BIASES_XXX**. 

The second block of parameters is specific to image segmentation. If the segmentation results are not satisfactory (i.e. organisms incompletely clipped, debris or other noise segmented as organisms, etc), changing these values might produce better results: 
.. # Image parsing specific

 - ``impa_image_format``: ``[jpg, png, ...]`` what format are the original images in? Should be caps insensitive and support common formats like ``jpg``, ``png`` and others. 
 - ``impa_clip_areas``: ``[int, int]`` it's common to place a reference scale and colour grid in images (see also :doc:`Introduction to examples <examples/ex_intro>`), here you can define the area of the image where it is placed, so that it can be cropped out. You should specify this as the coordinates (in pixels) of the bottom-right corner of the portion of the image you want analysed, so that the regions that fall outside of it can be cropped out, for example ``[2750, 4900]`` will exclude all areas *right* of 2750 pixles and *below* 4900 pixels. 
 - ``impa_area_threshold``: this is the minimum size (in pixels) that will be considered to be an organism; anything below this threshold will be discarded. When in doubt, start with a low threshold and increase until most noise is removed. 
 - ``impa_gaussian_blur``: ``[int, int]`` the size fo the kernel that will be used to smooth the image before processing; you can think of this as the "radius" of the blur: the larger the radius, the stronger the smoothing effect, but also more loss of details in the image. This should not be changed much except for very noisy images and/or with comparatively large organisms compared to the full size of the image. 
 - ``impa_gaussian_blur_passes``: ``[int]`` How many times the gaussian filter should be applied in sequence. 
 - ``impa_adaptive_threshold_block_size``: ``[int]`` Size of the square neighborhood used to collect values and statistics for automatic thresholding. 
 - ``impa_mask_postprocess_kernel``: ``[int, int]`` This is the size of the post-processing kernel, that smooths out the segmentation masks; higher values correspond to smoothers edges but less details. 
 - ``impa_mask_postprocess_passes``: ``[int]`` Number of times the smoothing kernel is applied. 
# impa_save_full_mask_dir: data/derived/project_portable_flume/full_image_masks
- ``impa_bounding_box_buffer``: ``[int]`` how many pixels should be added on each side of the mask for buffer (this is useful to evaluate if masks are accurate, for example). 
 - ``impa_save_clips_plus_features``: ``[bool]`` Boolean value (`True/False`) whether the features of each mask should be saved as CSV. 

This block contains parameters for model training and inferences. 
.. # Run classification routine on image clips 
.. ## Preparation of learning sets (run once if output folder is not there)
.. ## these data will need to be doctored, to move classes like errors 
.. ## and such into specific subfolers

 - ``lset_class_cut``: ``[kingdom, phylum, class, subclass, order, suborder, family, genus, species]`` This determines the taxonomic rank for cutoff, meaning that all lower taxonomic levels will be clumped together at the specified rank. Annotations at higher taxonomic level than the one specified will be excluded. 
 - ``lset_val_size``: ``[float]`` Which proportion of the annotated data should be set aside for validation? A common value is ``0.1``. 
 - ``lset_taxonomy``: ``[string]`` Full path to the location of the taxonomy file, for example: ``/data/mzb-workflow/data/MZB_taxonomy.csv``. This should be in ``.csv`` format, and contain a full taxonomy of the organisms in the annotated data (see `The taxonomy file`_). 

The following parameters relate to model training of the classification model. The proposed values will likely work for small datasets (<10'000 images) and a moderate number of classes (<20-30). Machine Learning (ML) model training is a complex topic, explanations given are very general and will likely be insufficient to fully grasp the all the intricacies! 
.. ## Finetuning / training config for classifier
 
 - ``trcl_learning_rate``: ``[float]`` This parameter controls the learning rate of the model; the higher the value the quicker it will adjust the weights, but also the quicker it will overfit. Suggested value: ``0.001``. 
 - ``trcl_batch_size``: ``[int]`` The number of images that will be used for training at each iteration. Higher numbers will use more memory and will achieve good accuracies faster, but small numbers will train the model faster. Suggested value: ``16``. 
 - ``trcl_weight_decay``: ``[float]`` How much should the weight of a node in the network decrease (i.e. decay) at each step (see ``trcl_step_size_decay``); decay combats overfitting but can slow down training. Suggested value: ``0``. 
 - ``trcl_step_size_decay``: ``[int]`` How many iterations before applying the weight decay factor. Suggested value: ``5``. 
 - ``trcl_number_epochs``: ``[int]`` How many iterations (i.e. epochs) should the model be trainer for. Longer training cycles can potentially yield better accuracies, but they take longer to train and can quickly overfit. Suggested value: ``75``. 
.. # trcl_gpu_ids: -1 
 
 - ``trcl_save_topk``: ``[int]`` How many models should be saved among the best? You can specify if you want to retain the best 1-2-5 etc best models after training; this can be beneficial for evaluating overfitting and convergence. Suggested value: ``1``. 
 - ``trcl_num_classes``: ``[int]`` How many classes should the model be trained for? This needs to be defined by the user, and it corresponds to how many taxa are at the specified taxonomic rank. In our example we had ``8``. 
 - ``trcl_model_pretrarch``: ``[convnext-small, resenet50, efficientnet-b2, convnext-small, densenet161, mobilenet]`` Which model architecture should be used for training; the supported architectures are detailed in :ref:`workflow_models:Available models` **NEED TO FIX THIS REFERENCE!** 
 - ``trcl_num_workers``: ``[int]`` How many processes (i.e. workers) do you want the dataloader to spawn? A good rule of thumb is to use the same number of workers as number of threads of your CPU. In our example the value is ``16``. 
 - ``trcl_wandb_project_name``: ``[string]`` Name of the Weights & Biases tracker for your project; you should change this to something meaningful for your project; in our case it was ``mzb-classifiers``. 
 .. # trai_model_save_append: "-v1"


.. ## Finetuning / training config for skeleton prediction

trsk_learning_rate: 0.0001
trsk_batch_size: 32
trsk_weight_decay: 0
trsk_step_size_decay: 50
trsk_number_epochs: 750
# trsk_gpu_ids: -1
trsk_save_topk: 1
trsk_num_classes: 2
trsk_model_pretrarch: mit_b2 #mit-b2 #efficientnet-b2
trsk_num_workers: 16
trsk_wandb_project_name: mzb-skeletons
# trsk_tversky_loss_w1: 
# trai_model_save_append: "-v1"

## Inference config 
# infe_model_folder: models/mzb-classifiers/ # likely not used to allow renku parse as input
infe_model_ckpt: last # best or last, best is on validation error
infe_num_classes: 8
infe_image_glob: "*_rgb.jpg" 

## Skeletonization
# unsupervised skeletonization
skel_class_exclude: errors
skel_conv_rate: 131.6625 #[133.1, 136.6, 133.2, 133.2, 133.2, 118.6, 133.4, 132.0])  # px / mm
# skel_save_usnup_masks: data/derived/project_portable_flume/skeletons/automatic_skeletons/

# supervised skeletonization
skel_label_thickness: 3
skel_label_buffer_on_preds: 25
skel_label_clip_with_mask: False # We need same set data (blobs and skeletonization 


Complete configuration file for *Portable Flume*
***********************************************

.. code-block:: yaml

    # Arguments not to be spec via CLI
    glob_random_seed: 222 
    glob_root_folder: /data/shared/mzb-workflow/
    glob_blobs_folder: /data/shared/mzb-workflow/data/derived/blobs/
    glob_local_format: pdf
    model_logger: wandb

    # Image parsing specific 
    impa_image_format: jpg
    impa_clip_areas: [2750, 4900] # ignore areas outside of this (bottom right corner)
    impa_area_threshold: 5000 # ignore areas smaller than this
    impa_gaussian_blur: [21, 21]
    impa_gaussian_blur_passes: 3
    impa_adaptive_threshold: 351
    impa_mask_postprocess_kernel: [11, 11]
    impa_mask_postprocess_passes: 5
    # impa_save_full_mask_dir: data/derived/project_portable_flume/full_image_masks
    impa_bounding_box_buffer: 200
    impa_save_clips_plus_features: True

    # Run classification routine on image clips 
    ## Preparation of learning sets (run once if output folder is not there)
    ## these data will need to be doctored, to move classes like errors 
    ## and such into specific subfolers
    lset_class_cut: order
    lset_val_size: 0.1
    lset_taxonomy: /data/shared/mzb-workflow/data/MZB_taxonomy.csv

    ## Finetuning / training config for classifier
    trcl_learning_rate: 0.001
    trcl_batch_size: 16
    trcl_weight_decay: 0
    trcl_step_size_decay: 5
    trcl_number_epochs: 75
    # trcl_gpu_ids: -1 
    trcl_save_topk: 1
    trcl_num_classes: 8
    trcl_model_pretrarch: convnext-small #resenet50 #efficientnet-b2 #convnext-small #densenet161 #mobilenet
    trcl_num_workers: 16
    trcl_wandb_project_name: mzb-classifiers
    # trai_model_save_append: "-v1"

    ## Finetuning / training config for skeleton prediction
    trsk_learning_rate: 0.0001
    trsk_batch_size: 32
    trsk_weight_decay: 0
    trsk_step_size_decay: 50
    trsk_number_epochs: 750
    # trsk_gpu_ids: -1
    trsk_save_topk: 1
    trsk_num_classes: 2
    trsk_model_pretrarch: mit_b2 #mit-b2 #efficientnet-b2
    trsk_num_workers: 16
    trsk_wandb_project_name: mzb-skeletons
    # trsk_tversky_loss_w1: 
    # trai_model_save_append: "-v1"

    ## Inference config 
    # infe_model_folder: models/mzb-classifiers/ # likely not used to allow renku parse as input
    infe_model_ckpt: last # best or last, best is on validation error
    infe_num_classes: 8
    infe_image_glob: "*_rgb.jpg" 

    ## Skeletonization
    # unsupervised skeletonization
    skel_class_exclude: errors
    skel_conv_rate: 131.6625 #[133.1, 136.6, 133.2, 133.2, 133.2, 118.6, 133.4, 132.0])  # px / mm
    # skel_save_usnup_masks: data/derived/project_portable_flume/skeletons/automatic_skeletons/

    # supervised skeletonization
    skel_label_thickness: 3
    skel_label_buffer_on_preds: 25
    skel_label_clip_with_mask: False # We need same set data (blobs and skeletonization training set are not the same filenames)

The taxonomy file
*****************
