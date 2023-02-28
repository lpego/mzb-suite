
# %% 
%load_ext autoreload
%autoreload 2 

import json
import zipfile
from pathlib import Path

import pandas as pd

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
    PLOTS = False
else:
    prefix = "../../"  # or "../"
    PLOTS = True


# %% 
root_annotations = Path(f"{prefix}data/2021_swiss_invertebrates/")

if not (root_annotations / "manual_measurements").is_dir():

    files_manual_phenopype = Path(f"{prefix}data/2021_swiss_invertebrates/manual_measurements-20230206T080300Z-001.zip")

    with zipfile.ZipFile(files_manual_phenopype, 'r') as zip_ref:
        zip_ref.extractall(root_annotations)

root_annotations = root_annotations / "manual_measurements"
# %%
measures = []
cols = ["species", "clip_name", "head", "body"]
files_to_merge = list(sorted(root_annotations.glob("**/*/*.json")))

for jfi in files_to_merge[:]: 

    clip_name = jfi.parent.name.split("__")[-1]
    species = clip_name.split("_")[2]

    with open(jfi) as f:
        data = json.load(f)

    body = data["line"]["body"]["data"]["lengths"]
    head = data["line"]["head"]["data"]["lengths"]

    measures.append(pd.DataFrame({"species": species, 
                                  "clip_name": clip_name, 
                                  "head_length": head,  
                                  "body_length": body}))
    
all_measures = pd.concat(measures)
all_measures.to_csv(root_annotations / "manual_annotations_summary.csv")
# %%
