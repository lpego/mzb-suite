import os
import time, datetime
import tkinter as tk
from tkinter import filedialog
import yaml
import streamlit as st

def select_folder(): 
    """
    Prompt user to select folder from filesystem
    """
    root = tk.Tk()
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_path

def file_selector(folder_path):
    """
    Button to list all files in selected folder and display in dropdown; 
    works in combination with select_folder.
    """
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

### Functions below require configuration by variables
### normally declared at the top of the streamlit page script
### Example below: 
# VARIABLES = {
#     "config_file": {
#         "type": "file",
#         "description": "Path to config file",
#         "two_stage": True,
#         "stage1_label": "Select folder for config file",
#         "stage2_label": "Select config file",
#     },
#     "input_dir": {"type": "folder", "description": "Directory containing the masks"},
#     "output_dir": {"type": "folder", "description": "Directory to save results"},
#     "save_masks": {"type": "folder", "description": "Directory to save masks as JPG"},
#     "list_of_files": {
#         "type": "file",
#         "description": "CSV file with classification predictions",
#         "two_stage": True,
#         "stage1_label": "Select folder for list of files",
#         "stage2_label": "Select list of files",
#     },
#     "verbose": {"type": "boolean", "description": "Verbose mode"},
# }
### Admissible types are "file", "folder", "boolean"
### When two_stage is True, file_selector is called after select_folder. 

def clear_session_state(VARIABLES):
    """
    Button to clear all user inputs from UI; forces app reload. 
    """
    for var in VARIABLES.keys():
        st.session_state[var] = None
        if VARIABLES[var].get("two_stage"):
            st.session_state[f"{var}_folder"] = None
    st.rerun()  # Refresh the UI after clearing

def validate_session_state(VARIABLES):
    """
    Checks that all required states have been assigned by user input, 
    otherwise returns an error and list of missing requirements. 
    """
    missing = [var for var in VARIABLES.keys() if var != "verbose" and not st.session_state.get(var)]
    if missing:
        st.write(f"**:red[Missing required inputs: {', '.join(missing)}]**")
        return False
    return True

def log_session_state(start_time, VARIABLES):
    """
    Returns YAML-like logs as dictionary for debug; saves started and end time of run tooo. 
    """
    streamlit_log = {
        "start_time": start_time,
        **{var: st.session_state[var] for var in VARIABLES.keys()},
        "end_time": datetime.datetime.now(),
    }
    return streamlit_log