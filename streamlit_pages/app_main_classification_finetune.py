import streamlit as st
import tkinter as tk
from tkinter import filedialog
import os
import time

# run the file: streamlit run app.py

st.title ('mzbsuite')
import os
import time, datetime
import tkinter as tk
from tkinter import filedialog
import streamlit as st
import yaml
from mzbsuite.streamlit.streamlit_utils import file_selector, select_folder

### Custom functions
def clear():
    st.session_state.config_file=False
    st.session_state.input_dir=False
    st.session_state.save_path=None


def checkEmpty():
   config_main_var=''
   input_main_var=''
   save_var=''

   try:
      config_main_var = st.session_state.config_file
   except:
      st.write('**:red[Please select path to config file with per-script args]**')

   try:
      input_main_var = st.session_state.input_dir
   except:
      st.write('**:red[Please select path with images for training]**')

   try:
      save_var = st.session_state.save_path
   except:
      st.write('**:red[Please select path to where to save model checkpoints]**')

   if  config_main_var != '' and input_main_var != '' and save_var != '':

      return True
   else:

      return False

debug = True # set to False for regular operation
streamlit_log = {"app_main_classification_finetune": {"start_time": datetime.datetime.now()}}

   
### Actual variables to grab
### Config file nested folder-file select
col1, col2 = st.columns([2, 4])
### Set up nested buttons states
if not "config_folder" in st.session_state:
    st.session_state["config_folder"] = False
### Select config folder
with col1: 
    config_folder = st.session_state.get("config_folder", None)
    # st.write("Select folder where config files are stored")
    config_folder_button = st.button("Select config folder")
    if config_folder_button:
        config_folder = select_folder()
        st.session_state.config_folder = config_folder
### Select file in config_folder
with col2:
    if st.session_state.config_folder is not False:
        config_file = file_selector(config_folder)
        st.session_state.config_file = config_folder
        st.write('You selected `%s`' % config_file)
        streamlit_log["app_main_classification_finetune"]["config_file"] = config_file


input_dir = st.session_state.get("input_dir", None)
input_button = st.button("Select path to learning sets (trn and val)")
if input_button:
  input_dir = select_folder()
  st.session_state.input_dir = input_dir

if input_dir:
   st.write("Selected folder path: `%s`" % input_dir)
   streamlit_log["app_main_classification_finetune"]["input_dir"] = input_dir

save_model = st.session_state.get("save_path", None)
save_button = st.button("Select path to where to save model checkpoints")
if save_button:
  save_model = select_folder()
  st.session_state.save_path = save_model

if save_model:
   st.write("Selected folder path: `%s`" % save_model)
   streamlit_log["app_main_classification_finetune"]["save_model"] = save_model

verbose = st.session_state.get("verbose", None)
verbose_checkbox = st.checkbox("Show more info")
if verbose_checkbox:
    verbose = True
else: 
    verbose = False

streamlit_log["app_main_classification_finetune"]["verbose"] = verbose

### Testing printouts
if debug: 
    st.write("DEBUGGING INFO:", streamlit_log)

# Finish button
loadingButton = st.button('Finish')

if loadingButton and checkEmpty():

    # Progress bar
    progress_text = 'Loading...'
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(9)
    my_bar.empty()

    ### Write the YAML
    streamlit_log["app_main_classification_finetune"]["end_time"] = datetime.datetime.now()
    with open('streamlit_log.yaml', 'a') as outfile:
        yaml.dump(streamlit_log, outfile, sort_keys=False)
    del(streamlit_log)
    st.write('Success, appending run parameters to "streamlit_log.yaml"')