Best practices
==============
Keep your workspace organised
----------------------------- 
The idea is that ``dataset : configuration : script`` should be ``1 : 1 : 1``:

 - Keep your raw data (i.e. full images) in ``/data/my_imgs``. 
 - Make a new configuration file with your settings in ``/configs`` by copying ``/configs/mzb_example_config.yaml`` and using it as a template. 
 - Make a new workflow file with your running parameters in ``/workflows``, using one of the examples as template. 
 - For your output folders, put your processed images in ``/data/my_imgs/derived``, and the model predictions in ``/results/my_imgs``. 
 - Name all the files related to one dataset in a similar way: ``configs/MZB_is_awesome_config.yml``, ``data/MZB_is_awesome``, ``data/derived/MZB_is_awesome`` and so on. 
 - If you change parameters, name your output folders differently so that you don’t mix the outputs of your experiments! 
 - If (re-)training ML models, it’s important to keep track of  (hyper)parameters and logs, see :ref:`files/best_practices:Logging your model's training`

.. hint:: 
    Read more about project organisation `here <https://drive.google.com/file/d/1W_Lq6JbqW8uSs3Y746xjRNWoGshiC1Y6/view>`__ 

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Logging your model's training
-----------------------------
To be able to tell whether a model is learning properly and/or is overfitting, it's necessary to log its progress while training. We support two loggers for this: 

 - For `Weights & Biases <https://docs.wandb.ai/>`__, you will need to create (free) account and install the necessary dependencies; refer to the documentation `here <https://wandb.ai/site/experiment-tracking>`__. After installing all requirements, run ``wandb login`` and provide your credentials when prompted.
 - For `TensorBoard <https://www.tensorflow.org/tensorboard>`__, please follow the installation instructions `here <https://www.tensorflow.org/tensorboard/get_started>`__. You will also need to specify which logger to use in the ``model_logger`` parameter in the configuration file (see :ref:`files/configuration:Configuration`). 