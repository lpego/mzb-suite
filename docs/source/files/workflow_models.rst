Workflow and Models
===================

We are going to write an to the workflow and models. 

Here also usage of workflows ``.sh`` files is shown.


Classificaiton
--------------


Available Models
~~~~~~~~~~~~~~~~

So far, these architectures are available for classificaiton: 

- ``vgg``: VGG 16
- ``resnet18``: ResNet 18 layers
- ``resnet50``: ResNet 50 layers
- ``densenet161``: DenseNet 161 layers
- ``mobilenet``: MobileNet V2 
- ``efficientnet-b2``: EfficientNet B2
- ``efficientnet-b1``: EfficientNet B1
- ``vit16``: Vision Transformer 16 
- ``convnext-small``: ConvNext Small

The models are pretrained on ImageNet and can be downloaded from the PyTorch model zoo. We use ``torchvision.models`` to load the models, and we pass ``weights=<ModelName>_Weiths.IMAGENET1K_V1`` for the pretrained weights. This can be changed depdending on needs.

Adding a new model
~~~~~~~~~~~~~~~~~~

In ``mzbsuite/utils.py`` you can add either a case to the function ``read_pretrained_model(architecture, n_class)`` or add a function returning a pytorch model. In general, the layers of these classifiers are all frozen and only the last fully connected layers are trained on the annotated data. This seemed to work in most of our cases, but can be changined in a simple way in the function. 

Skeleton Prediction
-------------------
In this toolbox, we offer two ways of performing and estimation of the size of the MZB. One is unsupervised, and relies on the estimation of the skeleton from the binary mask. The other is supervised, and relies on training a model from manually generated data. 

Automatic Skeleton Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main funtion ``scripts.skeletons.main_unsupervised_skeleton_estimation.py`` implements the unsupervised skeleton estimation from binary masks. We estimate the mask's skeleton, estimate a filament segmentation, and compute the longest path traversing the whole skeleton. We return the longest path, assuming it corresponds to the length of the MZB body.  

The algorithm is as follows and it is applied on every single binary mask of the MZB. Each mask is represented by 0 and 1 values, where 0 is the background and 1 is the MZB region.

1. The distance transform is computed using the function ``scipy.ndimage.distance_transform_edt``. We divide the mask in different predefined area classes, and use area-dependent parameters to threshold the distance transform. 
2. We select the larges area from the thresholded distance transform, as we assume this is the MZB main body.
3. We apply thinning using ``skimage.morphology.thin`` to the selected area.
4. We find intersections and endpoints of the skeleton using the custom implementation in ``mzbsuite.skeletons.mzb_skeletons_helpers.get_intersections`` and ``mzbsuite.skeletons.mzb_skeletons_helpers.get_endpoints``.
5. We compute the longest path using ``mzbsuite.skeletons.mzb_skeletons_helpers.traverse_graph``. We could test using other implementations. 
6. We save a CSV table at ``{out_dir}/skeleton_attributes.csv`` containing: 
    - ``clip_filename``: the name of the clip
    - ``conv_rate_mm_px``: the conversion rate from pixels to mm
    - ``skeleton_length``: the length of the skeleton in pixels
    - ``skeleton_length_mm``: the length of the skeleton in mm
    - ``segms``: the ID of the segmentation of the filaments (images with all filaments are stored in a defined folder)
    - ``area``: the area of the MZB in pixels (computed as the sum of the pixels in the binary mask)


Supervised Skeleton Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

