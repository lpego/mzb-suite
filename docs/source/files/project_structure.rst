Project structure
=================

The project contains several directories, but the code is stored in two main places: 

    #. The ``mzbsuite`` package, containing the main classes and functionalities; 
    #. The ``scripts`` and ``workflow`` directories making use of its functions. 

The ``mzbsuite`` folder  contains high-level functions that can be imported from other scripts and used to create complex processing pipelines (in Python this is called a package), while the ``scripts`` and ``workflow`` folders contain files that make use of functions implemented in ``mzbsuite`` to run single modules or the processing pipeline as a whole. 

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Directories and files
---------------------
Below we describe what the various directories in the project repository are used for. 

- ``data/``: is meant to contain the raw image data, and by the deafult the processed images (e.g. image clips, masks, skeletons, etc derived from the raw data) are stored there too, in a ``derived`` subdirectory; we can also store in ``data``  the taxonomy file (see :ref:`files/configuration:The taxonomy file`). 

    .. .. note:: 
    ..     **TODO** Some CSV with derived measures and features are momentarily stored there but might not always be needed. We can maybe generate those conditionally by addition option in ``configs/global_configuration.yaml``.

- ``mzbsuite/``: contains the functions for each module; these are called in the scripts provided but can also be called from users' own scripts by importing the ``mzbsuite`` package. 
- ``scripts/``: contains the scripts to run the different modules. Each module has its own folder, and each script is named after the module it implements. These modules can be run in isolation (i.e. each one can be run independently from the others from the command line). See :ref:`files/examples/Workflow files`, and also :ref:`files/workflow_models:Models` for details on the arguments these scripts take. 

    .. .. note:: 
    ..     **TODO** Need to make so that renku workflow can track main inputs and outputs without making it too complex. 

- ``models/``: contains the pretrained models used in the different modules. By default, new models are first downloaded from Pytorch model zoo, and the finetuned models are then saved in this folder. A few pre-trained models are provided. 
- ``configs/``: contains the project configuration file, ``config.yaml``, which contains all the parameters for the different modules. This file contains all the settings and hyperparameters for image processing and moel training, and it can be modified by the user to change the behavior of the scripts. The configuration files is always a required input to the scripts. 

    .. .. note:: 
    ..     **TODO** Maybe good idea to create copies of this config, with per-experiment naming, or create a branch of the repo, etc. Make sure to version those also, with good commit names. 

- ``results/``: is created by defualt upon running the scripts or workflows, and is meant to contain the outputs of the different modules. Each module has its own folder, and each script is named after the module it implements. 
- ``workflows/``: contains the workflows bash scripts (`.sh`), that allow to run one or multiple modules with shared parameters (e.g. on the same dataset). Running parameters are also declared in this bash scripts; refer to :ref:`files/examples/Workflow files` for a more detailed explanation. 

The file ``environment.yml`` contains all the minimal dependencies for the project, and should install the functions in ``mzbsuite`` as well. However, if this does not work, the ``mzbsuite`` package can be installed separately using the ``setup.py`` file in the ``mzbsuite`` folder, via ``pip`` (see :ref:`here <pip_install_mzbsuite>`)

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modules
-------
This project contains 3 modules: 

.. image:: ../../assets/MZB_pipeline_v2.1.png

#. Module :ref:`files/scripts/processing_scripts:Segmentation`: this module is mainly used as preprocessing steps for classification and measurements, but contains handy functionalities on its own. The raw input images are wide RGB photographs of many organisms, which are hard to process on their own. A first step is to detect all the organisms as independent and disjoint objects, and to crop them out of the original image. This is done with traditional image processing methods, and it is completely unsupervised. From this, we derive i) an RGB crop of the organism, with filename corresponding to the original image name plus clip ID and ii) a binary mask of the insect, filename corresponding to the original image name plus a clip ID. The binary mask can be used to compute the area of the insect (number of pixels) and RGB + clip can be used to compute local descriptors. 

#. Module :ref:`files/scripts/processing_scripts:Skeleton Prediction`: this module contains the code to measure the length and head width of organisms. The user can chose two approaches: completely unsupervised or supervised. The first, unsupervised approach uses image filtering, mathematical morphology and a graph-traversal algorithm to come up with a measure approximating only the length of the organisms, made from a the mask obtained from the original images; performance is better for organisms that are long and thin, slightly less accurate for complex shapes. The second approach to measure size is based on supervised Deep Learning (DL) models, which are trained based on manual annotations provided by a human expert of organism length (head to tail) and head width; the two predictions of the model correspond to the probability of each pixel to be along the "body" skeleton segments, or along the "head" skeleton segments. Postprocessing thins out those predictions to a line, and the size of the sample is then approximated by the sum of the length of the body and the head, respectively. The scripts in this module also allow users to finetune this DL model on their own data, and to use it for inference on new images. 

#. Module :ref:`files/scripts/processing_scripts:Classification`: this module contains the code to train and test a model to classify the organisms in MZB samples, according to a a set of predefined classes. Image classes in our case were encoded in the filename, but this not need be the case, as long as files are stored in a hierarchical directory structure. We also provide a pre-processing script to reclassify class names based on :ref:`files/configuration:The taxonomy file`, that groups classes according to taxonomic hierarchy. The data used to fine tune pretrained DL classifiers is then duplicated according to this hierarchical structure (each class in its own folder) in ``data/learning_sets/{project_name}/aggregated_learning_sets/``.