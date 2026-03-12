# %%
import os
from pathlib import Path

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
else:
    prefix = "../"  # or "../"

### Define the location for which you want to clean up file and directory names
ROOT_DIR = Path(f"/data/shared/mzb-suite/data/raw/dubendorf_ponds/October_2019")

# %% Step 1: make all paths lowercase, and ensure that " " are replaced by "_"
files_proc = list(ROOT_DIR.glob("**/*.*"))
files_proc.sort()
# files_proc

# %%  Step 2: parse also parent directories's names
for i, file_base in enumerate(files_proc[:]):
    print(f"{i+1}/{len(files_proc)}: {str(file_base)}")
    for i in range(4):
        if (".." == str(file_base.parents[i]))|("." == str(file_base.parents[i])):
            continue

        if file_base.parents[i].exists():
            os.rename(str(file_base.parents[i]), str(file_base.parents[i]).lower())
        # if file_base.parents[1].exists():
        #     os.rename(str(file_base.parents[i]), str(file_base.parents[i]).lower())
        # if file_base.parents[2].exists():
        #     os.rename(str(file_base.parents[i]), str(file_base.parents[i]).lower())

    sub_f = list(Path(str(file_base.parents[0]).lower()).glob("**/*.*"))

    for file_s in sub_f:
        os.rename(str(file_s), str(file_s).lower())

# %% Step 3: remove trailing spaces
for i, file_base in enumerate(files_proc[:]):
    print(f"{i+1}/{len(files_proc)}: {str(file_base)}", end=" ")

    # if " " in str(file_base):
    #     os.rename(str(file_s), str(file_s).replace(" ", "_"))
    #     print("**")
    # else:
    #     print()

# %% Step 4: remove all "*_mask.jpg" files, potential leftovers from previous scripts
ROOT_DIR = Path(f"{prefix}data/data_raw_custom_processing/")
files_proc = list(ROOT_DIR.glob("**/*_mask.jpg"))
files_proc.sort()

for i, file_base in enumerate(files_proc):
    os.remove(str(file_base))

# %%
