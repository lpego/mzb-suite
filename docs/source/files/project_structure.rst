Project structure
=================

The project contains two main parts: 

    #. The ``mzbsuite`` package, containing the main classes and functionalities; 
    #. The ``scripts`` and ``workflow`` directories making use of its functions. 

The ``mzbsuite`` folder  contains high-level functions that can be imported from other scripts and used to create complex processing pipelines (in Python this is called a package), while the ``scripts`` and ``workflow`` folders contain files that make use of functions implemented in ``mzbsuite`` to run single modules or the processing pipeline as a whole. 

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Directories and files
---------------------
Below we describe what the various directories in the project repository are for. 

- ``data/``: is meant to contain the raw image data, and the processed images (e.g. image clips, masks, skeletons, etc) that are derived from the raw data, as well as the taxonomy file (see :ref:`files/configuration:The taxonomy file`). 

    .. .. note:: 
    ..     **TODO** Some CSV with derived measures and features are momentarily stored there but might not always be needed. We can maybe generate those conditionally by addition option in ``configs/global_configuration.yaml``.

- ``mzbsuite/``: contains the functions for each module; these are called in the scripts provided but can also be called from users' own scripts by importing the ``mzbsuite`` package. 
- ``scripts/``: contains the scripts to run the different modules. Each module has its own folder, and each script is named after the module it implements. These modules can be run in isolation (i.e. each one can be run independently from the others from the command line). Arguments to these scripts are only the paths to the input and output folders, as well as the name of the project and model names (see :ref:`files/examples/Workflow files` also :ref:`files/workflow_models:Models`). Users need to specify the location of their own data and output folders on which to perform inference. Some work might be needed to ensure that they can also be run interactively, but for this, it is better to duplicate those files in a ``notebooks/`` folder, and to run them interactively from there after modding them. 

    .. .. note:: 
    ..     **TODO** Need to make so that renku workflow can track main inputs and outputs without making it too complex. 

- ``models/``: contains the pretrained models used in the different modules. The models are first downloaded from Pytorch model zoo, then finetuned. The finetuned models are then saved in this folder. A few pre-trained models are provided. 
- ``configs/``: contains the project configuration file, ``config.yaml``, which contains all the parameters for the different modules. This file contains all settings and hyperparameters, and it can be modified by the user to change the behavior of the scripts. The configuration files is always used as input to the scripts. 

    .. .. note:: 
    ..     **TODO** Maybe good idea to create copies of this config, with per-experiment naming, or create a branch of the repo, etc. Make sure to version those also, with good commit names. 

- ``results/``: is created upon running the scripts or workflows, and is meant to contain the results of the different modules. Each module has its own folder, and each script is named after the module it implements. 
- ``workflows/``: contains the renku workflows, for now contains just an implementation of the serial pipeline in bash scripts. One for the inference pipeline, and two for finetuning the classification and size measurement models.

The file ``environment.yml`` contains all the minimal dependencies for the project, and should install the functions in ``mzbsuite`` as well. However, if this does not work, the ``mzbsuite`` package can be installed separately using the ``setup.py`` file in the ``mzbsuite`` folder, via ``pip`` (see :ref:`here <pip_install_mzbsuite>`)

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modules
-------
This project contains 3 modules: 

.. image:: ../../assets/MZB_pipeline_v2.1.png

#. Module :ref:`files/scripts/processing_scripts:Segmentation`: this module is mainly used as preprocessing steps for classification and measurements, but contains handy functionalities on its own. The raw input images are wide RGB photographs of many MZB organisms, which are hard to process on their own. A first step is to detect all the insects as independent and disjoint objects, and to crop them out of the original image. This is done with traditional image processing methods, and it is completely unsupervised. From this, we derive i) an RGB crop of the organism, with filename corresponding to the original image name plus clip ID and ii) a binary mask of the insect, filename corresponding to the original image name plus a clip ID. The binary mask can be used to compute the area of the insect (number of pixels) and RGB + clip can be used to compute local descriptors. 

#. Module :ref:`files/scripts/processing_scripts:Skeleton Prediction`: this module contains the code to measure the length and width of organisms. The user can chose two approaches: completely unsupervised or supervised. The first, unsupervised approach uses image filtering, mathematical morphology and a graph-traversal algorithm to come up with a measure approximating *only the length* of the organisms, made from a the mask obtained from the original images; performance is better for samples that are long and thin animals, slightly less accurate for complex shapes. The second approach to measure size is based on supervised Deep Learning (DL) models, which are trained based on manual annotations provided by a human expert of insects length (head to tail) and head width (width of head); the two predictions of the model correspond to the probability of each pixel to be along the "body" skeleton segments, or along the "head" skeleton segments. Postprocessing thins out those predictions to a line, and the size of the sample is then approximated by the sum of the length of the body and the head, respectively. The scripts in this module also allow users to finetune this DL model on their own data, and to use it for inference on new images. 

#. Module :ref:`files/scripts/processing_scripts:Classification`: this module contains the code to train and test a model to classify the organisms in MZB samples, according to a a set of predefined classes. In our experiments, image classes were specified in the filename, but this can be changed if necessary. We reclassify class names thanks to the :ref:`files/configuration:The taxonomy file`, which groups classes according to taxonomic hierarchy. The data used to fine tune pretrained DL classifiers is then duplicated according to this hierarchical structure (each class in its own folder) in ``data/learning_sets/{project_name}/aggregated_learning_sets/``.