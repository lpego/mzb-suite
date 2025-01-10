.. mzb-suite documentation master file, created by
   sphinx-quickstart on Fri Jun 30 09:14:50 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================================================
MZB-WORFLOW: Macrozoobenthos data workflow suite 
================================================

Welcome to the ``mzb-suite`` documentation! Here we introduce the package functionality and illustrate some examples of the *Macrozoobenthos data processing workflow and suite* ``mzb-suite``. 

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quickstart
==========

To try out ``mzb-suite``, the quickest way to is to spin a virtual session on RenkuLab, follow instructions for :ref:`files/installing:Online session on RenkuLab` (only recommended for demo purposes). 

.. If you are not familiar with the JupyterLab interface of the virtual session, have a look at :ref:`files/examples/read_example:Working with notebooks`. 

.. In the Jupyter interface, to launch the example workflow and inspect the results, see :ref:`files/how_to_use:Workflow files`. 

If you want to install locally (recommended for day to day use), follow the instruction using :ref:`files/installing:Docker container` (recommended for novice users) or :ref:`files/installing:Install libraries locally` (advanced users). 

To see how you can run the project scrips head over to :doc:`files/how_to_use`. 

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MZB project
===============

:doc:`files/installing`
    How to install ``mzb-suite`` online or on your machine.

:doc:`files/how_to_use`
    How to work with the project, details about the workflows and its modules.

:doc:`files/project_structure`
    What does what in the project repository, and details about models used.

:doc:`files/configuration`
    Explanation fo the configuration file and recommended parameter values. 

:doc:`files/best_practices`
    Some tips on how to organise your project to make your life easier. 

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gallery
=======

:doc:`files/examples/demo`
    Demonstration of ``mzb-suite`` features. 

.. :doc:`_collections/files/examples/read_example`
..     How to change paths to files in notebooks

:doc:`_collections/files/examples/segmentation`
    Example of extracting clips from large images

:doc:`_collections/files/examples/classification_inference`
    Example of automatically identifying taxa from organisms clips

.. :doc:`_collections/files/examples/classification_finetune`
..     Example of retraining classification model

:doc:`_collections/files/examples/skeletonization_unsupervised`
    Example of extracting body length from organism clips

:doc:`_collections/files/examples/skeletonization_supervised_inference`
    Example of extracting body length and head width from organism clips

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scripts
=======

:doc:`files/scripts/processing_scripts`
    Detailed explanation for processing functions

:doc:`files/scripts/preprocessing`
    Details of other convenience scripts

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

API reference
=============

:doc:`files/modules/mzbsuite`
    Documentation on functions and parameters

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. Hidden TOCs

.. toctree::
    :caption: Project Documentation
    :maxdepth: 1
    :hidden:

    files/installing
    files/how_to_use
    files/project_structure
    files/configuration
    files/best_practices

.. toctree::
    :caption: Gallery
    :maxdepth: 1
    :hidden:

    files/examples/demo
    _collections/files/examples/segmentation
    _collections/files/examples/classification_inference
    _collections/files/examples/classification_finetune
    _collections/files/examples/skeletonization_unsupervised
    _collections/files/examples/skeletonization_supervised_inference

.. toctree::
    :caption: Scripts
    :maxdepth: 1
    :hidden:

    files/scripts/processing_scripts
    files/scripts/preprocessing

.. toctree::
    :caption: API reference
    :maxdepth: 1 
    :hidden:

    files/modules/mzbsuite