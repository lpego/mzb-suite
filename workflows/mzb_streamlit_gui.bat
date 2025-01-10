@echo off
@REM ## This script launches the mzbsuite GUI, based on streamlit ##
@REM ## Please specify your working directory (ROOT_DIR) below:  
SET ROOT_DIR=D:\mzb-workflow

@REM @REM Make sure Python can import modules
@REM python -i -m mzbsuite.utils

@REM Now run the Streamlit GUI
mamba activate mzbsuite_streamlit
streamlit run %ROOT_DIR%\streamlit_homepage.py

@REM ## Save this file, the run it via terminal from the appropriate conda/mamba environment. 