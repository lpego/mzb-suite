<!-- ![GitHub Pages](https://github.com/lpego/sphinx-rtd-test/actions/workflows/deploy_docs.yaml/badge.svg) -->
[![ReadTheDocs status](https://readthedocs.org/projects/mzb-suite/badge/?version=latest)](https://mzb-suite.readthedocs.io/en/latest/?badge=latest)
<!-- ![GitLab Docker build](https://gitlab.renkulab.io/biodetect/mzb-workflow/badges/master/pipeline.svg) -->

# What is mzb-suite? 
`mzb-suite` is an image processing pipeline for lab images of macrozoobenthos (MZB), partially automating data extraction from images.  

![Overview of mzb-suite](docs/assets/MZB_pipeline_v3.1.png)

## What can it be used for? 
- Segment individual MZB organisms from large-pane images into individual clips (unsupervised).  
- Extract total length of individual organisms (unsupervised), and both length and head width for selected taxa (supervised). 
- Use pre-trained Machine Learning (ML) models to predict coarse-grained identity for selected of taxa. 
- Assisted ML model re-training on other taxa using user-provided annotations. 
- Assists in organising data in a folder structure that is easy to navigate, and that can be used for further analysis. 
- Worked examples and documentation are provided to adapt the pipeline to users' projects. 

## Who can use it? 
Anyone that wants to process images of MZB or other organisms acquired in a lab setting, with fixed focal length, uniform background and lighting conditions. The pipeline can handle small amounts of noise in the images, but any debris similar in size to the organisms of interest will not be filtered out, making this pipeline unsuitable for images taken in the field. 

-------------------------------------------

# Get started
<!-- LINKS -->
You can get the trained models and some demo data here: https://doi.org/10.5281/zenodo.17581222

Please see the [documentation](https://mzb-workflow.readthedocs.io/en/latest/), it explains everything relating to the package. You can jump to directly to sections here: 

1. [Installation](https://mzb-workflow.readthedocs.io/en/latest/files/installing.html)

2. [Workflow and Models](https://mzb-workflow.readthedocs.io/en/latest/files/workflow_models.html)

3. [Examples](https://mzb-workflow.readthedocs.io/en/latest/files/examples/read_example.html)

4. [Processing scripts](https://mzb-workflow.readthedocs.io/en/latest/files/scripts/processing_scripts.html#)

5. [mzbsuite module reference](https://mzb-workflow.readthedocs.io/en/latest/files/modules/mzbsuite.html)

-------------------------------------------
# Project info

## Contributors & contacts
- Luca Pegoraro (WSL) - luca.pegoraro@wsl.ch
- Michele Volpi (SDSC) - mivolpi@ethz.ch

Full authors and contribution details in list in [this file](AUTHORS). 
 
## Issues & feature requests
If you encounter a reproducible bug, please prepare a MWE and open an Issue [here](https://gitlab.renkulab.io/biodetect/mzb-workflow/-/issues) where we can track it. 

Development time is limited for this project, so no major new features are being implemented at the moment. You are of course welcome to open a [pull request](https://gitlab.renkulab.io/biodetect/mzb-workflow/-/merge_requests), we will try to examine it quickly! 

## How to cite
_coming soon..._

## Changelog 

Find it [here](CHANGELOG). 
