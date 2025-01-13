import os
import time, datetime
import tkinter as tk
from tkinter import filedialog
import yaml

### streamlit utils
import streamlit as st
import mzbsuite.streamlit.streamlit_utils as stutils

### mzbsuite utils and scripts
from mzbsuite.utils import cfg_to_arguments, noneparse
from scripts.skeletonization.main_unsupervised_skeleton_estimation import main

# Configuration by variables
VARIABLES = {
    "config_file": {
        "type": "file",
        "description": "Path to config file",
        "two_stage": True,
        "stage1_label": "Select folder for config file",
        "stage2_label": "Select config file",
    },
    "input_dir": {"type": "folder", "description": "Directory containing the masks"},
    "output_dir": {"type": "folder", "description": "Directory to save results"},
    "save_masks": {"type": "folder", "description": "Directory to save masks as JPG"},
    "list_of_files": {
        "type": "file",
        "description": "CSV file with classification predictions",
        "two_stage": True,
        "stage1_label": "Select folder for list of files",
        "stage2_label": "Select list of files",
    },
    "verbose": {"type": "boolean", "description": "Verbose mode"},
}

# Initialize session state for all variables
for var, props in VARIABLES.items():
    if var not in st.session_state:
        st.session_state[var] = None

# Display fields based on config
# st.title("Unsupervised skeletonization estimation")
st.write("Please provide the following inputs:")

for var, props in VARIABLES.items():
    if props.get("two_stage"):  # Two-stage file selector
        col1, col2 = st.columns([2, 4])

        with col1:
            st.write(f"**{props['description']}**")
            if st.button(props["stage1_label"], key=f"{var}_folder_button"):
                st.session_state[f"{var}_folder"] = stutils.select_folder()
            if st.session_state.get(f"{var}_folder"):
                st.write(f"Folder selected: `{st.session_state[f'{var}_folder']}`")

        with col2:
            if st.session_state.get(f"{var}_folder"):
                st.write(f"**{props['stage2_label']} in `{st.session_state[f'{var}_folder']}`**")
                # Dropdown for file selection from the folder
                files = [f for f in os.listdir(st.session_state[f"{var}_folder"]) if os.path.isfile(os.path.join(st.session_state[f"{var}_folder"], f))]
                selected_file = st.selectbox(f"Select a file for {var}", files, key=f"{var}_file")
                if selected_file:
                    st.session_state[var] = os.path.join(st.session_state[f"{var}_folder"], selected_file)
                if st.session_state.get(var):
                    st.write(f"File selected: `{st.session_state[var]}`")

    else:  # Regular single-stage selectors
        st.write(f"**{props['description']}**")
        if props["type"] == "file":
            if st.button(f"Select {props['description']}", key=f"{var}_button"):
                st.session_state[var] = stutils.file_selector()
        elif props["type"] == "folder":
            if st.button(f"Select {props['description']}", key=f"{var}_button"):
                st.session_state[var] = stutils.select_folder()
        elif props["type"] == "boolean":
            st.session_state[var] = st.checkbox(props['description'])

        if st.session_state[var]:
            st.write(f"Selected: `{st.session_state[var]}`")

# Start time
start_time = datetime.datetime.now()

# Buttons: Finish and Clear
col1, col2 = st.columns(2)

with col1:
    if st.button("Finish"):
        if stutils.validate_session_state(VARIABLES):
            
            # args = {key: st.session_state[key] for key in st.session_state}
            args = stutils.log_session_state(start_time, VARIABLES) # just use the log dictionary
            args.pop("start_time", None) # remove unnecessary key:value pairs
            args.pop("end_time", None) # remove unnecessary key:value pairs
            args = cfg_to_arguments(args)
            # st.write(args)
            
            with open('D:\mzb-workflow\configs\mzb_example_config.yaml', "r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = cfg_to_arguments(cfg)
            # st.write(cfg)
            
            ### Finally, launch the script!
            main(args, cfg, st_GUI=True) # st_GUI flag that is launched from GUI
            
            # Save log
            streamlit_log = stutils.log_session_state(start_time, VARIABLES)
            with open("streamlit_log.yaml", "a") as outfile:
                yaml.dump(streamlit_log, outfile, sort_keys=False)

            st.write('**:green[Success!]** Run parameters saved to "streamlit_log.yaml".')
        else:
            st.write("**:red[Please complete all required inputs.]**")

with col2:
    if st.button("Clear"):
        stutils.clear_session_state(VARIABLES)  # Reset variables and refresh the UI

# Debugging info
if st.session_state.get("verbose"):
    st.write("DEBUG INFO:", {var: st.session_state[var] for var in VARIABLES.keys()})
