Installation
============

The Project contains two main parts: 
1. The ``mzbsuite`` package 
2. The scripts and workflow files making use and implementing their use. 

The ``mzbsuite`` contains higher-level functions that can be combined and used to create more complex processing pipelines, while the scripts and workflow files are used to implement the processing pipelines as a whole. 

The file ``environment.yml`` contains the all the minimal dependencies for the project, and should install ``mzbsuite`` as well. However, if this does not work, the ``mzbsuite`` package can be installed separately using the ``setup.py`` file in the ``mzbsuite`` folder, via ``pip``. 

Cloning the project
-------------------

The project is currently hosted on the `Swiss Data Science Center <https://datascience.ch>`__ GitLab server. To clone the project, you simply need to clone it into your projects folder: 

.. code-block:: bash

    git clone git@renkulab.io:biodetect/mzb-workflow.git


This will create a folder called ``mzb-workflow`` in your projects folder. You can then install the dependencies via conda and the ``environment.yml`` file: 

.. code-block:: bash

    cd mzb-workflow
    conda env create -f environment.yml


This should install ``mzbsuite`` as well, but if this does not work, you can simply install it via pip as: 

.. code-block:: bash

    pip install -e .

the ``-e`` flag will install the package in editable mode, so that you can make changes to the package and they will be reflected in your environment. 

