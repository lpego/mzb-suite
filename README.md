# What is mzb-suite? 
mzb-suite is a image processing pipeline for lab images of macrozoobenthos (MZB), partially automating data processing from images.  

![Overview of the mzb-suite](docs/assets/MZB_pipeline_v2.1.png)

## What can it be used for? 
- Segment individual macrozoobenthos organisms in images, both raw and manually annotated ones. 
- Organise data in a folder structure that is easy to navigate, and that can be used for further analysis. 
- Extract total length of individual organisms (unsupervised) and both length and head width (supervised). 
- Use pre-trained Machine Learning (ML) models to predict identity of a subset of taxa. 
- Assisted ML model re-training using user-provided annotations. 
- Worked examples and documentation are provided to adapt the pipeline to user project. 

## Who can use it? 
Anyone that wants to process images of macrozoobenthos acquired in a lab setting, with fixed focal length, uniform background and lighting conditions. The pipeline can handle small amounts of noise in the images, but any debris similar in size to the organisms of interest will not be filtered out, making this pipeline unsuitable for images taken in the field. 

-------------------------------------------

# Get started
<!-- LINKS -->

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

**v0.1.0** First release. Attempt to structure project and scripts in a way that is easy to understand and maintain.

## TODOs

_Not in any priority_ 
- [x] ALIGN ALL LEARNING SET IMAGES TO NEW PIPELINE, from pngs to jpgs
- [x] Check measures of supervised skeletonizations (length and width) and compare to manual annotations
- [x] LICENSE and AUTHORS and CITATION placeholders
- [x] update the skeleton files: image blobs are now named differently! 
- [x] Fix how save folders are passed for the supervised skeletonization

- [x] check env and pandas in it use, build `setupy.py` 
- [ ] Add module for evaluations, and for generating plots
- [ ] Add notebooks for plotting of results, images, etc. 
- [x] Check all docstrings and potenitally build documentation into html
- [ ] Add the dubendorf data use case to check consistency for multiple projects 
- [ ] Add a README.md to the data folder
- [x] Spend some time in thinking whether it is better to have one big config file, or one config file per module, or one config file per script.

- [x] check WANDBD accounts and api for loggers [Added support for tensorboard, running locally]
- [ ] Renku workflows for the different modules 
    - [Or add `renku run` ... in front of command in `sh` scripts]
