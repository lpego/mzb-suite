Workflow and Models
###################
Here we illustrate the main functionalities of the modules and information about the models. 
We also show the usage of ``workflows`` and its ``.sh`` files. 

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Introduction
============

This project contains 3 modules: 

#. Module :ref:`files/scripts/processing_scripts:Segmentation`: These are mainly used as preprocessing steps for classification and measurements, but are handy functionalities on their own. The raw input images are wide RGB photographs of many macrozoobenthos samples, which are hard to process on their own. A first step is to detect all the insects as independent and disjoint objects, and to crop them out of the original image. This is done with traditional image processing methods, and it is completely unsupervised. From this, we derive i) an RGB crop of the organism, with filename corresponding to the original image name plus clip ID and ii) a binary mask of the insect, filename corresponings to the original image name plus clip ID. The binary mask can be used to compute the area of the insect (number of pixels) and RGB + clip can be used to compute local descriptors. 

#. Module :ref:`files/scripts/processing_scripts:Skeleton Prediction`: this repo contains the code to measure the length and width of macrozoobenthos samples. The user can chose two approaches: one is completely unsupervised, and uses image filtering, mathematical morphology and graph-traversal algorithm to come up with a measure approximating *only the length* of the sample. Performance is better for samples that are long and thin, and slightly less accurate for complex shapes. The approximation for this approach is made from a binary mask obtained automatically from the MZB sample. The second approach to measure size is based on supervised deep learning models, which are trained based on some manual annotations of insects length (head to tail) and head width (width of head). Those annotations were provided by a human expert. The scripts in this repo allow to finetune this model, and to use this model for inference on new images. The two predictions of this model correspond to the probability of each pixel to be along the "body" skeleton segments, or along the "head" skeleton segments. Postprocessing thins those predictions to a line, and the size of the sample is then approximated by the sum of the length of the body and the head.

#. Module :ref:`files/scripts/processing_scripts:Classification`: this repo contains the code to train and test a model to classify the macrozoobenthos samples, according to a a set of predefined classes. In our experiments, image classes were specified in the filename, but this can be changed to a more flexible approach. We reclassify class names thanks to the ``data/MZB_taxonomy.csv`` file, which groups classes according to a taxonomic hierarchy. The data used to fine tune pretrained deep learning classifiers is then duplicated according to our structure (each class in its own folder) in ``data/learning_sets/project_portable_flume/aggregated_learning_sets/``.

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Working with the project
========================

Project structure
~~~~~~~~~~~~~~~~~

- ``data/``: is meant to contain the raw image data, and the processed images (e.g. image clips, masks, skeletons, etc) that are derived from the raw data, as well as the taxonomy file (see :ref:`files/configuration:The taxonomy file`). **TODO** Some CSV with derived measures and features are momentarily stored there but might not always be needed. We can maybe generate those conditionally by addition option in ``configs/global_configuration.yaml``.
- ``mzbsuite/``: contains the functions for each module; these are called in the scripts provided but can also be called from users' own scripts. 
- ``scripts/``: contains the scripts to run the different modules. Each module has its own folder, and each script is named after the module it implements. These modules can *be run in isolation* (i.e. each one can be run independently from the others from the command line). Arguments to these scripts are only the paths to the input and output folders, as well as the name of the project and model names (see also :ref:`files/workflow_models:Available models`). This is made so renku workflow can track main inputs and outputs (**TODO**) without making it too complex. Users just need to specify the location of their own data and output folders on which to perform inference.  Some work might be needed to ensure that they can also be run interactively, but for this, it is better to duplicate those files in a ``notebooks/`` folder, and to run them interactively from there after modding them. 
- ``models/``: contains the pretrained models used in the different modules. The models are first downloaded from pytorch model zoo, then finetuned. The finetuned models are then saved in this folder.
- ``configs/``: contains the project configuration file, ``config.yaml``, which contains all the parameters for the different modules. This file contains all settings and hyperparameters, and it can be modified by the user to change the behavior of the scripts. The configuration files is always used as input to ``main`` scripts. **TODO** Maybe good idea to create copies of this config, with per-experiment naming, or create a branch of the repo, etc. Make sure to version those also, with good commit names. 
- ``results/``: contains the results of the different modules. Each module has its own folder, and each script is named after the module it implements. It probably needs to be better structured and organized. 
- ``workflows/``: will contain the renku workflows, for now contains just an implementation of the serial pipeline in bash scripts. One for the inference pipeline, and two for finetuning the classification and size measurement models.

Workflow files
~~~~~~~~~~~~~~

In this folder, there are "workflow" files that can be used to run the pipeline. Those files are nothing else that a chain of python commands implementing the flow of the processing pipeline. For instance, just run ``./workflows/run_finetune_skeletonization.sh`` to fine tune the skeletonization model. 

    .. TODO: transfer those bash scripts to renku workflows, so that the renku can track the dependencies and the inputs and outputs of each step, and generate the graph of the workflow.

Simple parameters in these files (e.g. input and output folders), together with the parameters in the configuration file, control the execution of the scripts. 

Changing interactive session dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full conda env is given in ``environment.yml``. We need to check if the docker image builds...

Logging your model's training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To be able to tell whether a model is learning properly and/or is overfitting, it's necessary to log its progress while training. We support two loggers for this: `Weights & Biases <https://docs.wandb.ai/>`_ and `TensorBoard <https://www.tensorflow.org/tensorboard>`_. 

To be able to use Weights & Biases you will need to create (free) account and install the necessary dependencies; refer to the documentation here: 

- Weights & Biases: `<https://wandb.ai/site/experiment-tracking>`_

After installing all requirements, run ``wandb login``.

For TensorBoard, please follow the installation instructions here: 

- TensorBoard: `<https://www.tensorflow.org/tensorboard/get_started>`_

You will also need to specify which logger to use in the ``model_logger`` parameter in the configuration file (see :ref:`files/configuration:Configuration`). 

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Available models
================
So far, these architectures are available for classification: 

- ``vgg``: VGG 16
- ``resnet18``: ResNet 18 layers
- ``resnet50``: ResNet 50 layers
- ``densenet161``: DenseNet 161 layers
- ``mobilenet``: MobileNet V2 
- ``efficientnet-b2``: EfficientNet B2
- ``efficientnet-b1``: EfficientNet B1
- ``vit16``: Vision Transformer 16 
- ``convnext-small``: ConvNext Small

The models are pre-trained on ImageNet and can be downloaded from the PyTorch model zoo. We use ``torchvision.models`` to load the models, and we pass ``weights={ModelName}_Weigths.IMAGENET1K_V1`` for the pre-trained weights. This can be changed depending on needs. 

Adding a new model
------------------
In ``mzbsuite/utils.py`` you can either add a case to the function ``read_pretrained_model(architecture, n_class)`` or add a function returning a pytorch model. In general, the layers of these classifiers are all frozen and only the last fully connected layers are trained on the annotated data. This seemed to work in most of our cases, but can be changed in a simple way in the function. 
