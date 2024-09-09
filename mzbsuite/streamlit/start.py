import streamlit as st

st.set_page_config(
    page_title="mzb-suite",
)

st.write("# Welcome to mzb-suite")

st.sidebar.success("Select a task above.")

st.markdown(
    """
    mzb-suite is an image processing pipeline for lab images of macrozoobenthos (MZB), partially automating data extraction from images.
    **Select your task from the sidebar**
    # 
    # 
    #
    # 
    ### Resources and references
    - User Interface made with [streamlit.io](https://streamlit.io)
    - Inspired by [DeepMeerkat](http://benweinstein.weebly.com/deepmeerkat.html)
    # 
    # 
    ##### If you have questions contact [Luca.Pegoraro@WSL.ch](mailto:Luca.Pegoraro@WSL.ch)
    """
)