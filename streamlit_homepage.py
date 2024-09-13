import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

st.set_page_config(
    page_title="mzbsuite",
    # page_icon="🧊",   
    layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

st.logo("docs/assets/mzbsuite_logo_v2.1.svg")

st.sidebar.success("Select a task above.")

# If you want to use the no-sections version, this
# defaults to looking in .streamlit/pages.toml, so you can
# just call `get_nav_from_toml()`
nav = get_nav_from_toml(".streamlit/pages_sections.toml")

pg = st.navigation(nav)

add_page_title(pg)

pg.run()