Updates to ensure repo runs on new Renku v2.x

## New TODOs
- [ ] Change paths to Renku container locations
    - [ ] config files in `config`
    - [ ] hardcoded in some scripts
    - [ ] hardcoded in notebooks

- [ ] Make `docs\source\files\examples\demo.ipynb` actually run as an example
- [ ] Write tiny python script to unzip Zenodo archive in appropriate location

- [ ] Update documentation 
    - [ ] Launching a session in new Renku 
    - [ ] unzipping the demo data

- [ ] Double check local isntall
    - [ ] Have to remake Dockerfile from scratch (not relying on Renku)
    - [ ] Build and upload to (my personal) Dockerhub

### Extra
- [ ] Look into streamlit frontend that can be launched on Renku
    - [ ] Maybe offer also Docker container with streamlit frontend?

<!-- # ---------------------------------------------------------------------- # -->

# Old TODO list (from README)

### Features under consideration
- [ ] Add original image file name as a separate column in all `csv` outputs 
- [ ] Add options to provide multiple px/mm conversion rates, one for each image (i.e. parse `csv` file as dictionary to pass as arguments)
- [ ] Add module for evaluations, and for generating plots
- [x] Change scale exclusion parameter in `configs` & documentation so that you select square where scale is to exclude, instead of selecting the pixels to keep in the image. -- *changed config file and docs*

### General
- [x] Move taxonomy file location onto running parameters from configuration file
- [x] Clean unused workflow `sh` files and comments within them
- [x] `worflows/full_pipeline_custom.sh` has no execute permission in repo
- [x] Reduce images in example dataset
- [ ] ~~Set up push mirror GitLab → GitHub (without LFS files)~~
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
- [x] Cannot load correct kernel (with `mzbsuite`) in Jupyter notebooks from Renku interactive session... 
- [ ] ~~Finding and changing filepaths in Jupyter notebooks is difficult for users…~~

### Documentation
- [x] Put docs on ReadTheDocs and/or GitHub Pages
	- [x] Configure ReadTheDocs integration
	- [ ] ~~Configure automated docs build in GitHub Pages from mirror repo -- *unsupported*~~
	- [ ] ~~See if you can pull the commit name and reconstruct the docker image name on renku dynamically in the documentation (source/files/installing.rst)...~~ 
	- [x] Otherwise just grab a recent one that build correctly and stick with that
- [x] Add "Quickstart" section with tutorial in the documentation
- [x] in the documentation, `docs/source/files/workflow_models.rst`, add section called "Supervised Skeleton Prediction" and explain model architectures used for supevised skeleton prediction (this should also fix Sphinx build warnings for missing refs).
- [x] Merge ToDo in `README.md` and Evernote. 

### Notebooks 
- [ ] model retraining not working in notebook interactive environment... 
- [x] compress long code blocks? 
- [ ] direct links to documentation within markdown cells in notebooks 
- [x] make notebook for supervised skeletons finetuning 
	- [x] conflict with Jupyter notebook environment...
- [ ] Fix notebooks outputs: 
	- [x] `segmentation.ipynb` replace plots in-place while running instead of generating new ones. 
	- [x] `skeletonizatn_unsupervised.ipynb` replace plots in-place while running instead of generating new ones. 
	- [x] `skeletonization_supervised_inference.ipynb` returns empty predictions in notebook... 
	- [ ] `classification_finetune.ipynb` last cell (actually retraining the model) might not be compatible with an interactive environment...

### Figures to make for paper
- [x] Class (im)balance for flume MZB samples (classification and skeletons) 
- [x] Accuracy for classification model  - `results/project_portable_flume/class_convnext-small-v0_validation_set`
- [x] Accuracy for supervised skeletonization model (length and head width) - `results/project_portable_flume/skseg_mit-b2-v1_validation_set`
- [x] Accuracy for unsupervised skeletonization (length) 

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
