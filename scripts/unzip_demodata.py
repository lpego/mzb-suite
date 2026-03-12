# Source - https://stackoverflow.com/a/3451150
# Posted by Rahul, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-03, License - CC BY-SA 4.0

import os
from glob import glob
import zipfile
from urllib.request import urlretrieve
from tqdm import tqdm

demodata_paths = [
    "*demodata*.zip",
    "*/*demodata*.zip",
    "../*demodata*.zip",
    "../*/*demodata*.zip",
    "../../*/*demodata*.zip",
]

cwd = os.getcwd()
print("working in dir: ", f"{cwd}")

### Finding demo data .zip file
demodata = ""

for path in demodata_paths:
    # print(path)
    for filename in glob(path):
        # print(filename)
        if os.path.isfile(filename):
            print("Demo data archive found in: ", f"{filename}")
            demodata = filename
            break
    if demodata:
        break

### Downloading the archive if .zip file not found locally
if not demodata:
    print("Demo data archive not found locally, downloading...")
    url = "https://zenodo.org/records/17581223/files/mzbsuite_models_demodata.zip?download=1"
    demodata = os.path.join("..", "mzbsuite_models_demodata.zip")
    urlretrieve(url, demodata)
# print(demodata)

### Finding the root of the repo
repo_paths = [
    "../mzb-suite",
    "../*/mzb-suite",
    "../../mzb-suite",
    "../../*/mzb-suite",
]

repo = ""

for path in repo_paths:
    # print(path)
    for dir in glob(path):
        # print(dir)
        if os.path.isdir(dir):
            print("Repo root dir found in: ", f"{dir}")
            repo = dir
            break
    if repo:
        break
# print(repo)

### Unzipping demo data
with zipfile.ZipFile(demodata, 'r') as zip_ref:
    for member in tqdm(zip_ref.infolist(), desc="Extracting "):
        try:
            zip_ref.extract(member, repo)
        except zipfile.error as e:
            pass

print("Extraction complete.")