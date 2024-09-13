import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

st.set_page_config(layout="wide", page_title="mzb-suite")

st.write("# Welcome to mzb-suite!")

# If you want to use the no-sections version, this
# defaults to looking in .streamlit/pages.toml, so you can
# just call `get_nav_from_toml()`
nav = get_nav_from_toml(".streamlit/pages_sections.toml")

st.logo("docs/assets/mzbsuite_logo_v2.1.svg")

pg = st.navigation(nav)

# st.sidebar.success("Select a task above.")

add_page_title(pg)

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

pg.run()