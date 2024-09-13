import os
import time, datetime
import tkinter as tk
from tkinter import filedialog
import streamlit as st
import yaml

### streamlit utils
from mzbsuite.streamlit.streamlit_utils import file_selector, select_folder

### mzbsuite utils and scripts
from mzbsuite.utils import cfg_to_arguments, noneparse
from scripts.image_parsing.main_raw_to_clips import main

st.title ('Segmentation: main_raw_to_clips')

### Custom functions
# def clear():
#     st.session_state.config_file=False
#     st.session_state.input_dir=False
#     st.session_state.output_dir=None
#     st.session_state.save_full_mask_dir=None

def checkEmpty():
   config_main_var=''
   input_main_var=''
   output_main_var=''
   save_full_mask_dir_var=''
   try:
      config_main_var = st.session_state.config_file
   except:
      st.write('**:red[Please select path to config file with per-script args]**')
   try:
      input_main_var = st.session_state.input_dir
   except:
      st.write('**:red[Please select path to directory with raw images]**')
   try:
      output_main_var = st.session_state.output_dir
   except:
      st.write('**:red[Please select path to where to save model checkpoints]**')
   try:
      save_full_mask_dir_var = st.session_state.save_full_mask_dir
   except:
      st.write('**:red[Please select path to directory where to save labeled full masks]**')
   if  config_main_var != '' and save_full_mask_dir_var !='' and input_main_var != '' and output_main_var != '':
      return True
   else:
      return False

debug = True # set to False for regular operation

### Initialise YAML log
streamlit_log = {"app_main_raw_to_clips": {"start_time": datetime.datetime.now()}}
   
### Actual variables to grab
# Config file nested folder-file select
col1, col2 = st.columns([2, 4])
# Set up nested buttons states
if not "config_folder" in st.session_state:
    st.session_state["config_folder"] = False
# Select config folder
with col1: 
    config_folder = st.session_state.get("config_folder", None)
    st.write("Select folder where config files are stored")
    config_folder_button = st.button("Select config folder")
    if config_folder_button:
        config_folder = select_folder()
        st.session_state.config_folder = config_folder
# Select file in config_folder
with col2:
    if st.session_state.config_folder is not False:
        config_file = file_selector(config_folder)
        st.session_state.config_file = config_folder
        st.write('You selected `%s`' % config_file)
        streamlit_log["app_main_raw_to_clips"]["config_file"] = config_file

### Grab the other running parameters
input_dir = st.session_state.get("input_dir", None)
input_button = st.button("Select path to directory with raw images")
if input_button:
  input_dir = select_folder()
  st.session_state.input_dir = input_dir

if input_dir:
   st.write("Selected folder path: `%s`" % input_dir)
   streamlit_log["app_main_raw_to_clips"]["input_dir"] = input_dir

output_dir = st.session_state.get("output_dir", None)
save_button = st.button("Select path to where to save the segmented clips of organisms")
if save_button:
  output_dir = select_folder()
  st.session_state.output_dir = output_dir

if output_dir:
   st.write("Selected folder path: `%s`" % output_dir)
   streamlit_log["app_main_raw_to_clips"]["save_model"] = output_dir

save_full_mask_dir = st.session_state.get("save_full_mask_dir", None)
input_button = st.button("Select path to directory where to save labeled full masks")
if input_button:
  save_full_mask_dir = select_folder()
  st.session_state.save_full_mask_dir = save_full_mask_dir

if save_full_mask_dir:
   st.write("Selected folder path: `%s`" % save_full_mask_dir)
   streamlit_log["app_main_raw_to_clips"]["save_full_mask_dir"] = save_full_mask_dir

verbose = st.session_state.get("verbose", None)
verbose_checkbox = st.checkbox("Show more info")
if verbose_checkbox:
    verbose = True
else: 
    verbose = False
st.session_state.verbose = verbose
streamlit_log["app_main_raw_to_clips"]["verbose"] = verbose

### Debug info
if debug: 
   st.write("DEBUGGING INFO:", streamlit_log)

### Launch button
loadingButton = st.button('Launch!')

if loadingButton and checkEmpty():
   ### Create dictionary for argparse
   arg_list = ["input_dir",
               "output_dir",
               "save_full_mask_dir", 
               "verbose"
               ]
   args = {}
   for x in arg_list: 
      args[x] = eval(x)
   
   ### Convert to arguments
   args = cfg_to_arguments(args)
   
   with open(str(config_file), "r") as f:
      cfg = yaml.load(f, Loader=yaml.FullLoader)
      
   cfg = cfg_to_arguments(cfg)

   ### Finally, launch the script!
   main(args, cfg, st_GUI=True) # st_GUI flag that is launched from GUI

   ### Write out the YAML logs
   streamlit_log["app_main_raw_to_clips"]["end_time"] = datetime.datetime.now()
   with open('streamlit_log.yaml', 'a') as outfile:
      yaml.dump(streamlit_log, outfile, sort_keys=False)
   del(streamlit_log) # empty the log, ready for next iteration
   st.write('Success! Appending run parameters to "streamlit_log.yaml"')