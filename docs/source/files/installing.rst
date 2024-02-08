Installation
============

There are a few ways to work with the project: 

    #. Run it in an `Online session on RenkuLab`_ (easiest setup, but limited computing power)
    #. Run it on your local machine using the `Docker container`_ (medium setup, can use your computer resources)
    #. `Install libraries locally`_ directly on your computer (longest setup, troubleshooting potentially needed, but affords maximum flexibility)

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Online session on RenkuLab
---------------------------
To try out ``mzbsuite``, the quickest way to is to spin a virtual session on Renkulab: 

#. Go to `<https://renkulab.io/>`__ and create an account (or log in if you already have one). 
#. Once you're logged in, go to `<https://renkulab.io/projects/biodetect/mzb-workflow>`__ 
#. Start the virtual session: 

    #. in the project overview you should see a green play button
    #. click the dorp-down menu arrow on its right and select "Start with options"; 
    #. Under "Session class" select Large (1 CPU, 4 GB RAM, 64 GB disk)
    #. Drag the bar for "Amount of storage" to 10 GB. 
    #. Toggle the switch "Automatically fetch LFS data"
    #. At the bottom of the page click "Start session"
    #. Be patient! It can take several minutes for the session to start. 

.. image:: ../../assets/Renkulab_virtual_session.gif

Once the session has started you should drop directly into the JupyterLab interface. 
From there you can work through the notebooks in :ref:`Examples`. 

We don't recommend this method for for long term use, as the online session are shut down for maintenance, so data not exported will be lost. 

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Docker container
----------------
If you want to run the pipeline on yur own data and local computer resources but you are not too familiar with managing python environments, we recommend installing via Docker. 

#. First, install the `Docker Engine <https://docs.docker.com/get-docker/>`__ 

    .. admonition:: \ \ 

        Admin privileges may be needed to install Docker Engine.  

#. After the installation, launch the Docker app and wait until it's *fully* started. 

#. Open a terminal in a directory you would like to save your work in (you must have write permissions). 

    .. hint:: 

        If you don't know how to open a terminal, instructions for `Windows <https://superuser.com/a/340051>`__ and `Mac <https://support.apple.com/guide/terminal/open-new-terminal-windows-and-tabs-trmlb20c7888/mac>`__. 

#. Download and launch the project's Docker image by running the following lines in a terminal window: 

    ⚠️ WARNING: this may take a while to run the first time, as it needs to download the Docker image (~5 GB)

    .. code-block:: bash 
        
        imageName=registry.renkulab.io/biodetect/mzb-workflow:8805a38
        repoName=mzb-workflow
        docker run --rm -ti -v ${PWD}:/work/${repoName} --workdir /work/${repoName} -p 8888:8888 ${imageName} jupyter lab --ip=0.0.0.0

    .. hint:: 
        If you are on Windows, use this instead: 
        
        .. code-block:: console
            
            set imageName=registry.renkulab.io/biodetect/mzb-workflow:8805a38
            set repoName=mzb-workflow
            docker run --rm -ti -v %cd%:/work/%repoName% --workdir /work/%repoName% -p 8888:8888 %imageName% jupyter lab --ip=0.0.0.0
    
#. The last few lines of the output should look something like this: 

    .. code-block:: console

        To access the server, open this file in a browser:
            file:///home/jovyan/.local/share/jupyter/runtime/jpserver-14-open.html
        Or copy and paste one of these URLs:
            http://a62c488a6f4c:8888/lab?token=2fb162adc7251f04e37cb8d6f1f55db2fbdbc7d2e1d9e1e8
            http://127.0.0.1:8888/lab?token=2fb162adc7251f04e37cb8d6f1f55db2fbdbc7d2e1d9e1e8

#. Copy-paste the URL address starting with ``http://127.0.0.1`` into your browser (e.g. Firefox, Chrome), hit Enter, and you should drop into the JupyterLab interface. 

The procedure is the same when starting the Docker image the following times, except it will not download it again therefore should be much faster launching it. 

    .. hint:: 
        If you would rather use an IDE, we recommend `Visual Studio Code <https://code.visualstudio.com/>`__; to access the Docker container read about `VS Code Docker extension <https://code.visualstudio.com/docs/containers/overview>`__ and `working with containers <https://code.visualstudio.com/docs/devcontainers/containers>`__. 

Using the most up to date image
_______________________________
The Docker image linked above in ``imageName`` is a manually pinned image that we tested and we know works. It might not contain the most most up to date versions of scripts if we pushed changes to the repo recently. If you want to pull the most updated Docker image, please follow instructions `<here https://renku.readthedocs.io/en/stable/how-to-guides/own_machine/reusing-docker-images.html>`__. 

.. admonition:: \ \ 

    Make sure that the Docker image is correctly built before pulling it; you can check build status `here <https://gitlab.renkulab.io/biodetect/mzb-workflow/-/pipelines>`__. 

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install libraries locally
-------------------------

If you prefer to install the project directly in your local environment or just want to use the functions in your own scripts, you can download the project's  repository. 
The project is currently hosted on the `Swiss Data Science Center <https://datascience.ch>`__ GitLab server, you can find the repository here: 

    .. rst-class:: center

    `<https://gitlab.renkulab.io/biodetect/mzb-workflow>`__. 

To download the project, you simply need to clone it into a location of your choice: 

.. code-block:: bash

    git clone https://gitlab.renkulab.io/biodetect/mzb-workflow.git

.. admonition:: \ \ 
   
   If you don't have Git installed, you can follow instructions `here <https://git-scm.com/downloads>`__. We recommend using Git because it allows to easily update the package and tracking any changes you make. 

This will create a folder called ``mzb-workflow`` in the current working directory. 

.. hint:: \ \ 
   If you don't want to use Git, you can directly download an archive of the `repository <https://renkulab.io/gitlab/biodetect/mzb-workflow>`__ from GitLab and extract it manually. 

You can then install the necessary packages using the conda package manager and the ``environment.yml`` file: 

.. code-block:: bash

    cd mzb-workflow    # chdir mzb-workflow in Windows
    conda env create -f environment.yml

.. hint:: \ \ 
   If you don't have the ``conda`` command available, you need to install it.    
   We strongly recommend using Mamba, find installation instruction `here <https://mamba-framework.readthedocs.io/en/latest/installation_guide.html>`__; but if you prefer you can also install `Anaconda <https://www.anaconda.com/download>`__.

.. _pip_install_mzbsuite:

This should install the ``mzbsuite`` package as well, but if this does not work, you can simply install it via pip as: 

.. code-block:: bash

    pip install -e .

the ``-e`` flag will install the package in editable mode, so that you can make changes to the functions in ``mzbsuite`` and they will be reflected in your environment. 

.. admonition:: \ \ 
   
   You can check whether ``mzbsuite`` has been installed by running: 

   .. code-block:: bash
    
    conda list -n mzbsuite

   and check that ``mzbsuite`` appears in the  list. 

If there are no errors then you're all set up and can start using the modules! 