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
    st.session_state.input_dir=False
    st.session_state.manual_annotations=None
    st.session_state.model_annotations=None

def checkEmpty():
   input_main_var=''
   manual_annotations_var=''
   model_annotations_var=''

   try:
      input_main_var = st.session_state.input_dir
   except:
      st.write('**:red[Please select path to the directory with the model predictions]**')

   try:
      manual_annotations_var = st.session_state.manual_annotations
   except:
      st.write('**:red[Please select path to the manual annotations]**')

   try:
      model_annotations_var = st.session_state.model_annotations
   except:
      st.write('**:red[Please select path to the model predictions]**')

   if  model_annotations_var !='' and input_main_var != '' and manual_annotations_var != '':

      return True
   else:

      return False

debug = True # set to False for regular operation
streamlit_log = {"main_supervised_skeleton_assessment": {"start_time": datetime.datetime.now()}}

### Actual variables to grab
input_dir = st.session_state.get("input_dir", None)
input_button = st.button("Select path to the directory with the model predictions")
if input_button:
  input_dir = select_folder()
  st.session_state.input_dir = input_dir

if input_dir:
   st.write("Selected folder path: `%s`" % input_dir)
   streamlit_log["main_supervised_skeleton_assessment"]["input_dir"] = input_dir

manual_annotations = st.session_state.get("manual_annotations", None)
manual_button = st.button("Select path to the manual annotations")
if manual_button:
  manual_annotations = select_folder()
  st.session_state.manual_annotations = manual_annotations

if manual_annotations:
   st.write("Selected folder path: `%s`" % manual_annotations)
   streamlit_log["main_supervised_skeleton_assessment"]["manual_annotations"] = manual_annotations

model_annotations = st.session_state.get("model_annotations", None)
input_button = st.button("Select path to the model predictions")
if input_button:
  model_annotations = select_folder()
  st.session_state.model_annotations = model_annotations

if model_annotations:
   st.write("Selected folder path: `%s`" % model_annotations)
   streamlit_log["main_supervised_skeleton_assessment"]["model_annotations"] = model_annotations

verbose = st.session_state.get("verbose", None)
verbose_checkbox = st.checkbox("Show more info")
if verbose_checkbox:
    verbose = True
else: 
    verbose = False

streamlit_log["main_supervised_skeleton_assessment"]["verbose"] = verbose

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
    time.sleep(4.4)
    my_bar.empty()

    ### Write the YAML
    streamlit_log["main_supervised_skeleton_assessment"]["end_time"] = datetime.datetime.now()
    with open('streamlit_log.yaml', 'a') as outfile:
        yaml.dump(streamlit_log, outfile, sort_keys=False)
    del(streamlit_log)
    st.write('Success, appending run parameters to "streamlit_log.yaml"')