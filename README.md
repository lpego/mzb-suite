<!-- ![GitHub Pages](https://github.com/lpego/sphinx-rtd-test/actions/workflows/deploy_docs.yaml/badge.svg) -->
[![ReadTheDocs status](https://readthedocs.org/projects/mzb-workflow/badge/?version=latest)](https://mzb-workflow.readthedocs.io/en/latest/?badge=latest)
![GitLab Docker build](https://gitlab.renkulab.io/biodetect/mzb-workflow/badges/master/pipeline.svg)

# What is mzb-suite? 
`mzb-suite` is an image processing pipeline for lab images of macrozoobenthos (MZB), partially automating data extraction from images.  

![Overview of mzb-suite](docs/assets/MZB_pipeline_v2.1.png)

## What can it be used for? 
- Segment individual MZB organisms in images, both raw and manually annotated ones. 
- Extract total length of individual organisms (unsupervised), and both length and head width (supervised). 
- Use pre-trained Machine Learning (ML) models to predict identity of a subset of taxa. 
- Assisted ML model re-training using user-provided annotations. 
- Organize data in a folder structure that is easy to navigate, and that can be used for further analysis. 
- Worked examples and documentation are provided to adapt the pipeline to user project. 

## Who can use it? 
Anyone that wants to process images of MZB acquired in a lab setting, with fixed focal length, uniform background and lighting conditions. The pipeline can handle small amounts of noise in the images, but any debris similar in size to the organisms of interest will not be filtered out, making this pipeline unsuitable for images taken in the field. 

-------------------------------------------

# Get started
<!-- LINKS -->
The documentation is on [ReadTheDocs](https://mzb-workflow.readthedocs.io/en/latest/) 
<!-- and [GitHub Pages](https://lpego.github.io/mab-workflow/).  -->

1. Installation 

2. Workflow and Models

3. Examples

4. Processing scripts

-------------------------------------------

## Contributions
<!-- AUTHORS -->

<!-- CONTACTS -->

<!-- FEATURE REQUESTS?  -->

<!-- ### How to cite
LEAVE BLANK FOR PREPRINT OR PAPER
------------------------------------------- -->

## Changelog 

**v0.2.1** Adding CI to GitHub and GitHub Pages for serving docs. 

**v0.2.0** Pipeline in functional state, lots of modifications from previous ver...  

**v0.1.0** First release. Attempt to structure project and scripts in a way that is easy to understand and maintain.

## ToDo list

### Feature requests
- [ ] Add original image file name as a separate column in all `csv` outputs 
- [ ] Add options to provide multiple px/mm conversion rates, one for each image (i.e. parse `csv` file as dictionary to pass as arguments)
- [ ] Add module for evaluations, and for generating plots
- [x] Change scale exclusion parameter in `configs` & documentation so that you select square where scale is to exclude, instead of selecting the pixels to keep in the image. *changed config file and docs*

### General
- [x] Move taxonomy file location onto running parameters from configuration file
- [x] Clean unused workflow `sh` files and comments within them
- [x] `worflows/full_pipeline_custom.sh` has no execute permission in repo
- [x] Reduce images in example dataset
- [ ] Set up push mirror GitLab → GitHub (without LFS files)
- [ ] ~~Renku workflows for the different modules (or add `renku run` ... in front of command in `sh` scripts) → not sure about this one (yet)~~

### Functionality and data
- [x] Double check that excluding the millimetre/colour scale in images works properly *in cv2, x and y are flipped compared to most other implementations!*
- [ ] Add the Dübendorf data use case to check consistency for multiple projects? 
- [ ] Make example dataset for supervised skeletonization model finetuning
- [ ] Update `workflows/run_finetune_skeletonization.sh` accordingly

### JupyterLab via Renku
- [x] JupyterLab screws up all the filepaths… Working dir is home/jovyan/work
- [x] Notebooks not picking up correct conda env in JupyterLab... 
- [x] Cannot run .sh script in the console directly if in cwd??? 
- [x] Notebooks don’t pick up conda env… 
- [ ] Cannot load correct kernel (with `mzbsuite`) in Jupyter notebooks from Renku interactive session... 
- [ ] Finding and changing filepaths in Jupyter notebooks is difficult for users… 

### Documentation
- [x] Put docs on ReadTheDocs and/or GitHub Pages
	- [x] Configure ReadTheDocs integration
	- [ ] Configure automated docs build in GitHub Pages from mirror repo
	- [ ] ~~See if you can pull the commit name and reconstruct the docker image name on renku dynamically in the documentation (source/files/installing.rst)...~~ 
	- [x] Otherwise just grab a recent one that build correctly and stick with that
- [x] Add "Quickstart" section with tutorial in the documentation
- [x] in the documentation, `docs/source/files/workflow_models.rst`, add section called "Supervised Skeleton Prediction" and explain model architectures used for supevised skeleton prediction (this should also fix Sphinx build warnings for missing refs).
- [x] Merge ToDo in `README.md` and Evernote. 

### Notebooks 
- [ ] model retraining not working in notebook interactive environment... 
- [ ] compress long code blocks? 
- [ ] direct links to documentation within markdown cells in notebooks 
- [ ] make notebook for supervised skeletons finetuning 
- [ ] Fix notebooks outputs: 
	- [ ] `segmentation.ipynb` replace plots in-place while running instead of generating new ones. 
	- [ ] `skeletonizatn_unsupervised.ipynb` replace plots in-place while running instead of generating new ones. 
	- [ ] `skeletonization_supervised_inference.ipynb` returns empty predictions in notebook... 
	- [ ] `classification_finetune.ipynb` last cell (actually retraining the model) might not be compatible with an interactive environment...

### Figures to make for paper
- [x] Class (im)balance for flume MZB samples (classification and skeletons) 
- [x] Accuracy for classification model  - `results/project_portable_flume/class_convnext-small-v0_validation_set`
- [x] Accuracy for supervised skeletonization model (length and head width) - `results/project_portable_flume/skseg_mit-b2-v1_validation_set`
- [ ] Accuracy for unsupervised skeletonization (length) 

   
--- 
   
**OLD COMPLETED TODO ITEMS - v0.1.0** - _Not in any priority_ 
- [x] ALIGN ALL LEARNING SET IMAGES TO NEW PIPELINE, from pngs to jpgs
- [x] Check measures of supervised skeletonizations (length and width) and compare to manual annotations
- [x] LICENSE and AUTHORS and CITATION placeholders
- [x] update the skeleton files: image blobs are now named differently! 
- [x] Fix how save folders are passed for the supervised skeletonization
- [x] check env and pandas in it use, build `setupy.py` 
- [x] Add notebooks for plotting of results, images, etc. 
- [x] Check all docstrings and potentially build documentation into html
- [x] Add a README.md to the data folder
- [x] Spend some time in thinking whether it is better to have one big config file, or one config file per module, or one config file per script.
- [x] check `wandb` accounts and api for loggers (added support for tensorboard, running locally)
