import copy
import hashlib
import traceback
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import streamlit_ext as ste
from datetime import datetime as dt
from datetime import timedelta
from datetime import time as dt_time
import plotly.express as px
import plotly.graph_objs as go
import time
import urllib.parse
import requests, json
import pydeck as pdk
import geopandas as gpd
from shapely import wkb
import os
import plotly.express as px

from lib.lib_data_ingest import *
from lib.lib_utils import *
from lib.lib_conf import *

from page.tutorial_page import tutorial_page
from page.import_page import import_page
from page.input_page import input_page
from page.login_page import login_page
from page.results_page import results_page
from page.setting_page import setting_page
from page.query_history_page import query_history_page



# ptvsd.enable_attach(address=('localhost', 5678))


# dashboard setup
st.set_page_config(
    page_title="Real-Time Apple-Watch Heart-Rate Monitoring Dashboard",
    page_icon="🏥",
    layout="wide",
)

def main():
    # dashboard title
    session = st.session_state
    createNav()

    # Display the appropriate page based on the session state
    if session is None:
        st.error("Please run the app first.")
        return
    if session.get("page") == "tutorial":
        tutorial_page()
    elif session.get("login-state", False) == False or session.get("page", "login") == "login":
        login_page()
    elif session.get("page") == "input":
        # if session doesn't contain key "current_db"
        if not session.get("current_db"):
            session["current_db"] = getCurrentDbByUsername(session.get("login-username"))
        # show a drop list to choose current db

        pre_current_db = session.get('current_db')
        exist_databases = [""] + get_existing_databases()
        session["current_db"] = st.selectbox("Select a database", exist_databases, index=exist_databases.index(
            pre_current_db) if pre_current_db in exist_databases else 0)
        if pre_current_db != session.get('current_db'):
            pre_current_db = session.get('current_db')
            updateCurrentDbByUsername(session.get("login-username"), session.get('current_db'))
            if 'selected_users' in session.keys():
                del session['selected_users']
            st.experimental_rerun()

        if(session["current_db"] != ""):
            # garmin_df = get_garmin_df(get_db_engine(mixed_db_name=session["current_db"]))
            # garmin_df.age = garmin_df.age.astype(int)
            # garmin_df.weight = garmin_df.weight.astype(int)
            # garmin_df.height = garmin_df.height.astype(int)
            # input_page(garmin_df)
            input_page(config=load_config('conf/config.yaml'))
    elif session.get("page") == "import":
        import_page()
    elif session.get("page") == "results":
        results_page(config=load_config('conf/config.yaml'))
    elif session.get("page") == "query_history":
        query_history_page()
    elif session.get("page") == "setting":
        setting_page()



if __name__ == '__main__':
    if not st.session_state.get("page"):
        st.session_state['page'] = 'tutorial'
    main()