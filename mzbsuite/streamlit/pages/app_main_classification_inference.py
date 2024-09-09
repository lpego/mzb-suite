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
    st.session_state.taxonomy_file=None
    st.session_state.input_dir=False
    st.session_state.input_model=False
    st.session_state.output_dir=None


def checkEmpty():
   config_main_var=''
   taxonomy_file_var=''
   input_main_var=''
   input_model_var=''
   output_main_var=''

   try:
      config_main_var = st.session_state.config_file
   except:
      st.write('**:red[Please select path to config file with per-script args]**')

   try:
      taxonomy_file_var = st.session_state.taxonomy_file
   except:
      st.write('**:red[Please select path to csv with taxonomy of classes]**')

   try:
      input_main_var = st.session_state.input_dir
   except:
      st.write('**:red[Please select path with images for training]**')

   try:
      input_model_var = st.session_state.input_model
   except:
      st.write('**:red[Please select path to images for inference]**')

   try:
      output_main_var = st.session_state.output_dir
   except:
      st.write('**:red[Please select path to where to save classificaiton predictions as csv]**')

   if  config_main_var != '' and taxonomy_file_var !='' and input_main_var != '' and input_model_var != ''  and output_main_var != '':

      return True
   else:

      return False

debug = True # set to False for regular operation
streamlit_log = {"app_main_classification_inference": {"start_time": datetime.datetime.now()}}

   
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
        streamlit_log["app_main_classification_inference"]["config_file"] = config_file

col1, col2 = st.columns([2, 4])
### Set up nested buttons states
if not "taxonomy_folder" in st.session_state:
    st.session_state["taxonomy_folder"] = False
### Select config folder
with col1: 
    taxonomy_folder = st.session_state.get("taxonomy_folder", None)
    # st.write("Select folder where config files are stored")
    taxonomy_folder_button = st.button("Select path to csv with taxonomy of classes")
    if taxonomy_folder_button:
        taxonomy_folder = select_folder()
        st.session_state.taxonomy_folder = taxonomy_folder
### Select file in taxonomy_folder
with col2:
    if st.session_state.taxonomy_folder is not False:
        taxonomy_file = file_selector(taxonomy_folder)
        st.session_state.taxonomy_file = taxonomy_folder
        st.write('You selected `%s`' % taxonomy_file)
        streamlit_log["app_main_classification_inference"]["taxonomy_file"] = taxonomy_file


input_dir = st.session_state.get("input_dir", None)
input_button = st.button("Select path with images to perform inference on")
if input_button:
  input_dir = select_folder()
  st.session_state.input_dir = input_dir

if input_dir:
   st.write("Selected folder path: `%s`" % input_dir)
   streamlit_log["app_main_classification_inference"]["input_dir"] = input_dir

input_model = st.session_state.get("input_model", None)
input_button = st.button("Select path to model checkpoint for inference")
if input_button:
  input_model = select_folder()
  st.session_state.input_model = input_model

if input_model:
   st.write("Selected folder path: `%s`" % input_model)
   streamlit_log["app_main_classification_inference"]["input_model"] = input_model

output_dir = st.session_state.get("output_dir", None)
save_button = st.button("Select path to where to save classificaiton predictions as csv")
if save_button:
  output_dir = select_folder()
  st.session_state.output_dir = output_dir

if output_dir:
   st.write("Selected folder path: `%s`" % output_dir)
   streamlit_log["app_main_classification_inference"]["save_model"] = output_dir

verbose = st.session_state.get("verbose", None)
verbose_checkbox = st.checkbox("Show more info")
if verbose_checkbox:
    verbose = True
else: 
    verbose = False

streamlit_log["app_main_classification_inference"]["verbose"] = verbose

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
    time.sleep(4)
    my_bar.empty()

    ### Write the YAML
    streamlit_log["app_main_classification_inference"]["end_time"] = datetime.datetime.now()
    with open('streamlit_log.yaml', 'a') as outfile:
        yaml.dump(streamlit_log, outfile, sort_keys=False)
    del(streamlit_log)
    st.write('Success, appending run parameters to "streamlit_log.yaml"')