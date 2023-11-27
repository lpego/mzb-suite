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

To try out ``mzb-suite``, the quickest way to is to spin a virtual session on RenkuLab, follow instructions on :ref:`files/installing:Online session on RenkuLab`. 

If you are not familiar with the JupyterLab interface, have a look at :ref:`files/examples/read_example:Working with notebooks`. 

To launch the example workflow and inspect the results, see :ref:`files/workflow_models:Workflow files`. 

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MZB project
===============

:doc:`files/installing`
    How to install the project and ``mzb-suite`` package on your machine.

:doc:`files/project_structure`
    What does what in the project repository.

:doc:`files/workflow_models`
    How to work with the project, details about the workflows and models used.

:doc:`files/configuration`
    Explanation fo the configuration file and recommended parameter values. 

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examples
========

:doc:`files/examples/read_example`
    How to change paths to files in notebooks

:doc:`files/examples/segmentation`
    Example of extracting clips from large images

:doc:`files/examples/skeletonization_unsupervised`
    Example of extracting body length from organism clips

:doc:`files/examples/skeletonization_supervised_inference`
    Example of extracting body length and head width from organism clips

:doc:`files/examples/classification_inference`
    Example of automatically identifying taxa from organisms clips

:doc:`files/examples/classification_finetune`
    Example of retraining classification model

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Processing scripts
==================

:doc:`files/scripts/processing_scripts`
    Detailed explanation for processing functions

:doc:`files/scripts/diverse_preprocessing`
    Details of other convenience scripts

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. Hidden TOCs

.. toctree::
    :caption: Project Documentation
    :maxdepth: 1
    :hidden:

    files/installing
    files/project_structure
    files/workflow_models
    files/configuration

.. toctree::
    :caption: Examples
    :maxdepth: 1
    :hidden:

    files/examples/read_example
    files/examples/segmentation
    files/examples/skeletonization_unsupervised
    files/examples/skeletonization_supervised_inference
    files/examples/classification_inference
    files/examples/classification_finetune

.. toctree::
    :caption: Processing scripts
    :maxdepth: 1
    :hidden:

    files/scripts/processing_scripts
    files/scripts/diverse_preprocessing

.. toctree::
    :caption: mzbsuite Module
    :maxdepth: 1 
    :hidden:

    files/modules/mzbsuite