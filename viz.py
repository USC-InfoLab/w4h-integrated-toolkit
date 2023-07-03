import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, date, timedelta
import plotly.express as px
import plotly.graph_objs as go
import time
import random, hashlib
from sqlalchemy import create_engine, select
import urllib.parse
import requests, json
import ptvsd
from loguru import logger
import pydeck as pdk
import jinja2
from ipywidgets import HTML
import math



# ptvsd.enable_attach(address=('localhost', 5678))

from conf import *

# DEFAULT_START_DATE = date.today()
ACTIVITIES_REAL_INTERVAL = 15
ALERT_TIMEOUT = 60
DEFAULT_WINDOW_SIZE = 60
DEFAULT_MIN_HRATE = 60
DEFAULT_MAX_HRATE = 115

# Define the USC location as a latitude and longitude
USC_CENTER_Y = 34.0224
USC_CENTER_X = -118.2851


# get db engine
def get_db_engine():
    db_pass_enc = urllib.parse.quote_plus(DB_PASS)
    return create_engine(f'postgresql://{DB_USER}:{db_pass_enc}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# get user ids
def get_garmin_user_id(db_conn, pattern=None):
    # query = f"SELECT user_id FROM {DB_USER_TABLE} WHERE device LIKE '%Garmin%'"
    query = f"SELECT user_id FROM {DB_USER_TABLE}"
    params = []
    if pattern:
        query += f" WHERE subj_id LIKE %s"
        pattern = f'%{pattern}%'
        params = [pattern]
    # execute the query
    return pd.read_sql(query, db_conn, params=params).values.squeeze()

# get full user info
def get_garmin_df(db_conn, pattern=None):
    # query = f"SELECT * FROM {DB_USER_TABLE} WHERE device LIKE '%Garmin%'"
    query = f"SELECT * FROM {DB_USER_TABLE}"
    params = []
    if pattern:
        query += f" WHERE subj_id LIKE %s"
        pattern = f'%{pattern}%'
        params = [pattern]
    # execute the query
    return pd.read_sql(query, db_conn, params=params)


def calculate_mets(cal_df, user_weights=None):
    # if not user_weights:
    #     user_weights = dict(zip(cal_df.user_id.unique(), np.ones(cal_df.user_id.nunique()) * 70))
    # mets_df = cal_df.copy()
    # mets_df['value'] = mets_df.apply(lambda x: x['value'] / (user_weights[x['user_id']] * 0.25), axis=1)

    # return mets_df
    return pd.DataFrame(columns=['user_id', 'timestamp', 'value'])



# dashboard setup
st.set_page_config(
    page_title="Real-Time Apple-Watch Heart-Rate Monitoring Dashboard",
    page_icon="🏥",
    layout="wide",
)



# Flask server API endpoint
SERVER_URL = f"http://{HOST}:{PORT}"

# read data from Flask server
def get_data() -> pd.DataFrame:
    response = requests.get(SERVER_URL)
    data = response.json()
    df_hrate = pd.DataFrame(data)
    # df_hrate = pd.DataFrame(data['heart_rates'])
    # df_calories = pd.DataFrame(data['calories'])
    # df_coords = pd.DataFrame(data['coordinates'])
    df_hrate['timestamp'] = pd.to_datetime(df_hrate['timestamp'])
    # df_calories['timestamp'] = pd.to_datetime(df_calories['timestamp'])
    # df_coords['timestamp'] = pd.to_datetime(df_coords['timestamp'])
    df_hrate = df_hrate.set_index('timestamp')
    # df_calories = df_calories.set_index('timestamp')
    # df_coords = df_coords.set_index('timestamp')
    # return df_hrate, df_calories, df_coords
    return df_hrate, pd.DataFrame(columns=['user_id', 'timestamp', 'value']), pd.DataFrame(columns=['user_id', 'timestamp', 'value'])


def post_message_to_slack(text, blocks = None):
    return requests.post('https://slack.com/api/chat.postMessage', {
        'token': slack_token,
        'channel': slack_channel,
        'text': text,
        'icon_emoji': slack_icon_emoji,
        'username': slack_user_name,
        'blocks': json.dumps(blocks) if blocks else None
    }).json()


def get_control_stats(df_hrate_all, df_calories_all, df_mets_all, control_ids):
    df_hrate = df_hrate_all.query('user_id in @control_ids')
    df_calories = df_calories_all.query('user_id in @control_ids')
    df_mets = df_mets_all.query('user_id in @control_ids')
    stats = dict()
    stats['heart_rate'] = {'max': df_hrate.value.max(), 'min': df_hrate.value.min(),
                           'avg': df_hrate.value.mean(), 'std': df_hrate.value.std()}
    stats['calories'] = {'max': df_calories.value.max(), 'min': df_calories.value.min(),
                            'avg': df_calories.value.mean(), 'std': df_calories.value.std()}
    stats['mets'] = {'max': df_mets.value.max(), 'min': df_mets.value.min(),
                            'avg': df_mets.value.mean(), 'std': df_mets.value.std()}
    return stats


def add_aux_rectangles(fig, df, df_full, window_start, window_end):
    fig.add_shape(
        type='rect',
        xref='x', yref='paper',
        x0=window_start, y0=0,
        x1=window_end, y1=1,
        fillcolor='blue',
        opacity=0.1,
        layer='below',
        line_width=0
    )

    # calculate the avg and std of the feature. Define safe range as +-2 std away from the mean
    avg_val = df_full.value.mean()
    std_val = df_full.value.std()
    safe_min = avg_val - 2 * std_val
    safe_max = avg_val + 2 * std_val
    
    fig.add_shape(
        type='rect',
        xref='paper', yref='y',
        x0=0, y0=safe_min,
        x1=1, y1=safe_max,
        fillcolor='green',
        opacity=0.1,
        layer='below',
        line_width=0
    )

    for user_id, group in df.groupby('user_id'):
        unsafe_values = group[(group['value'] < safe_min) | (group['value'] > safe_max)]
        if not unsafe_values.empty:

            for i, unsafe_value in unsafe_values.iterrows():
                    fig.add_shape(
                        type='rect',
                        xref='x', yref='paper',
                        x0=i - timedelta(seconds=10), y0=0,
                        x1=i + timedelta(seconds=10), y1=1,
                        fillcolor='red',
                        opacity=0.2,
                        layer='below',
                        line_width=0
                    )


def get_bar_fig(df, label='Feature'):
    fig = px.bar(
                x=df.columns.tolist(),
                y=df.values.flatten().tolist()
    )

    fig.update_layout(
        width=250,
        height=300,
        showlegend=False,
        xaxis_title=None,
        yaxis_title=label,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    fig.update_traces(marker_color=['#636EFA', '#EF553B'])

    return fig

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)


def get_map_legend(color_lookup):
    # map_legend_lookup = [{'text': t, 'color': rgb_to_hex(c)} for t, c in color_lookup.items()]
    # legend_markdown = "<br>".join([f"<span style='color:{leg['color']}'> &#9679; </span>{leg['text']}" for leg in map_legend_lookup])
    # return st.markdown(legend_markdown, unsafe_allow_html=True)
    map_legend_lookup = [{'text': t, 'color': rgb_to_hex(c)} for t, c in color_lookup.items()]
    legend_markdown = "  \n".join([f"<span style='color:{leg['color']}'> &#9679; </span>{leg['text']}" for leg in map_legend_lookup])
    return st.markdown(f"<p style='font-size: 16px; font-weight: bold;'>Map Legend</p>{legend_markdown}", unsafe_allow_html=True)


# Define function to create Pydeck layer
def create_layer(df, color):
    coordinates = df['coordinates'].values.tolist()[0]

    layer = pdk.Layer(
        # 'user',
        type="PathLayer",
        data=df,
        pickable=True,
        get_color=color,
        auto_highlight=True,
        width_scale=20,
        width_min_pixels=2,
        get_path="coordinates",
        get_width=2
    )
    
    # define the ScatterplotLayer using the first coordinate
    marker_layer_start = pdk.Layer(
        "ScatterplotLayer",
        data=[{"position": coordinates[0]}],
        get_position="position",
        get_radius=20,
        get_fill_color=[0, 0, 255],
        pickable=True
    )
    
    marker_layer_end = pdk.Layer(
        "ScatterplotLayer",
        data=[{"position": coordinates[-1]}],
        get_position="position",
        get_radius=20,
        get_fill_color=[255, 0, 0],
        pickable=True
    )
    
    return [layer, marker_layer_start, marker_layer_end]


# Define the input page
def input_page(garmin_df):
    global TIMEOUT
    # Get the session state
    session = st.session_state
    if session is None:
        st.error("Please run the app first.")
        return
    
    # preparing data
    user_ids = garmin_df.user_id.tolist()
    rank_options = garmin_df['rank'].unique().tolist()
    drop_type_options = garmin_df['drop_type'].unique().tolist()
    weight_min, weight_max = int(garmin_df.weight.min()), int(garmin_df.weight.max())
    height_min, height_max = int(garmin_df.height.min()), int(garmin_df.height.max())
    age_min, age_max = int(garmin_df.age.min()), int(garmin_df.age.max())
        
    # top-level filters
    
    # Selecting the Subjects
    st.header("Select Subject(s)")
    # add selector for user
    subject_selection_options = ['id', 'attribute']
    subject_selection_type = st.radio("Select subject(s) by id or by attribute?", subject_selection_options, index=session.get('subject_selection_type', 0))
    
    selected_users = []
    if subject_selection_type == 'id':
        selected_users = st.multiselect(
            "Select Subject ID(s)",
            options=user_ids,
            default=[])
        
    selected_rank = []
    selected_drop_type = []
    selected_weight_range = []
    selected_height_range = []
    selected_age_range = []
    
    if subject_selection_type == 'attribute':
        st.subheader("Select Subject(s) Attributes")
        col1, col2, col3, col4, col5 = st.columns(spec=[1, 3, 3, 3, 1], gap='large')
        # add radio selector for gender
        selected_rank = col1.multiselect(
            "Select military rank",
            options=rank_options,
            key='subject rank',
            # index=session.get('selected_rank', 0)
            )
        selected_rank = selected_rank if selected_rank else rank_options

        # add sliders for weight, height, age
        selected_age_range = col2.slider(
            "Select age range (years)",
            min_value=age_min,
            max_value=age_max,
            value=session.get('selected_age_range', (age_min, age_max)),
            step=1,
            key='subject age',
            )
            
        selected_weight_range = col3.slider(
            "Select weight range (lbs)",
            min_value=weight_min,
            max_value=weight_max,
            value=session.get('selected_weight_range', (weight_min, weight_max)),
            step=1,
            key='subject weight')

        selected_height_range = col4.slider(
            "Select height range (inches)",
            min_value=height_min,
            max_value=height_max,
            value=session.get('selected_height_range', (height_min, height_max)),
            step=1,
            key='subject height')
        
        selected_drop_type = col5.multiselect(
            "Select drop type",
            options=drop_type_options,
            key='drop type',
            )
        selected_drop_type = selected_drop_type if selected_drop_type else drop_type_options
            
            
    # Selecting the control group
    st.header("Select Control Group")
    # add selector for user
    control_selection_options = ['all', 'id', 'attribute']
    control_selection_type = st.radio("Select control group (either as all studied individuals or filter by id or attribute)?", 
                                      control_selection_options,
                                      index=session.get('control_selection_type', 0))
    
    selected_users_control = []
    if control_selection_type == 'id':
        selected_users_control = st.multiselect(
            "Select Control Target ID(s)",
            options=user_ids,
            default=[])
        
    selected_rank_control = []
    selected_drop_type_control = []
    selected_weight_range_control = []
    selected_height_range_control = []
    selected_age_range_control = []
    
    if control_selection_type == 'attribute':
        st.subheader("Select Control Group Attributes")
        col1, col2, col3, col4, col5 = st.columns(spec=[1, 3, 3, 3, 1], gap='large')
        # add radio selector for gender
        selected_rank_control = col1.multiselect(
            "Select military rank",
            options=rank_options,
            key='control military rank',
            # index=session.get('selected_rank_control', 0)
            )
        selected_rank_control = selected_rank_control if selected_rank_control else rank_options

        # add sliders for weight, height, age
        selected_age_range_control = col2.slider(
            "Select age range (years)",
            min_value=age_min,
            max_value=age_max,
            value=session.get('selected_age_range_control', (age_min, age_max)),
            step=1,
            key='control age')
            
        selected_weight_range_control = col3.slider(
            "Select weight range (lbs)",
            min_value=weight_min,
            max_value=weight_max,
            value=session.get('selected_weight_range_control', (weight_min, weight_max)),
            step=1,
            key='control weight')

        selected_height_range_control = col4.slider(
            "Select height range (inches)",
            min_value=height_min,
            max_value=height_max,
            value=session.get('selected_height_range_control', (height_min, height_max)),
            step=1,
            key='control height')
        
        selected_drop_type_control = col5.multiselect(
            "Select drop type",
            options=drop_type_options,
            key='control drop type',
            # index=session.get('selected_rank_control', 0)
            )
        selected_drop_type_control = selected_drop_type_control if selected_drop_type_control else drop_type_options


    st.header("Visualization/Analysis Configuration")

    real_time_update = st.checkbox("Real-Time update?", value=True)

    if not real_time_update:
        start_date = st.date_input(
        "Start date",
        datetime.strptime(START_TIME, '%Y-%m-%d %H:%M:%S'))
        
        end_date = st.date_input(
        "End date",
        datetime.strptime(END_TIME, '%Y-%m-%d %H:%M:%S'))
    else:
        st.info("Real-Time update is enabled. Start/End dates for querying are disabled.")

    window_size = st.number_input('Window Size (seconds)', value=session.get("window_size", DEFAULT_WINDOW_SIZE), step=15)
    if real_time_update:
        TIMEOUT = st.number_input('Fast Forward (Every 15s Equals)', value=float(TIMEOUT), step=float(1), format="%.1f", min_value=0.1, max_value=float(100))
    

        
    # Add a button to go to the results page
    if st.button("Show Results"):
        
        # save input values to the session state
        session["window_size"] = window_size
        session["real_time_update"] = real_time_update
        session['subject_selection_type'] = 0 if subject_selection_type == 'id' else 1
        session['control_selection_type'] = 0 if control_selection_type == 'all' else 1 if control_selection_type == 'id' else 2
        session['selected_rank'] = selected_rank
        session['selected_rank_control'] = selected_rank_control
        session['selected_drop_type'] = selected_drop_type
        session['selected_drop_type_control'] = selected_drop_type_control
        session['selected_age_range'] = selected_age_range
        session['selected_age_range_control'] = selected_age_range_control
        session['selected_weight_range'] = selected_weight_range
        session['selected_weight_range_control'] = selected_weight_range_control
        session['selected_height_range'] = selected_height_range
        session['selected_height_range_control'] = selected_height_range_control
        
        
        # Filter the dataframe based on the selected criteria for subjects
        if subject_selection_type == 'id':
            subjects_df = garmin_df.query('user_id in @selected_users')
        else:
            subjects_df = garmin_df.query('rank == @selected_rank and drop_type == @selected_drop_type and weight >= @selected_weight_range[0] and weight <= @selected_weight_range[1] and height >= @selected_height_range[0] and height <= @selected_height_range[1] and age >= @selected_age_range[0] and age <= @selected_age_range[1]')
            
        # Filter the dataframe based on the selected criteria for control group
        if control_selection_type == 'all':
            control_df = garmin_df
        elif control_selection_type == 'id':
            control_df = garmin_df.query('user_id in @selected_users_control')
        else:
            control_df = garmin_df.query('rank == @selected_rank_control and drop_type == @selected_drop_type_control and weight >= @selected_weight_range_control[0] and weight <= @selected_weight_range_control[1] and height >= @selected_height_range_control[0] and height <= @selected_height_range_control[1] and age >= @selected_age_range_control[0] and age <= @selected_age_range_control[1]')
        
        # Store the filtered dataframe in session state
        session["subjects_df"] = subjects_df
        session["control_df"] = control_df
        
        # Go to the results page
        session["page"] = "results"
        st.experimental_rerun()


# Define the results page
def results_page():
    # Get the session state
    session = st.session_state
    if session is None:
        st.error("Please run the app first.")
        return
    
   
    subjects_df = session.get('subjects_df')
    subject_ids = subjects_df.user_id.tolist()
    control_df = session.get('control_df')
    control_ids = control_df.user_id.tolist()
    
    window_size = session['window_size']
    real_time_update = session['real_time_update']

    
    if 'df_hrate_full' not in st.session_state or 'df_calories_full' not in st.session_state or 'df_coords_full' not in st.session_state:
        st.session_state['df_hrate_full'] = pd.DataFrame()
        st.session_state['df_calories_full'] = pd.DataFrame()
        st.session_state['df_coords_full'] = pd.DataFrame()
        
    # Set initial view state
    view_state = pdk.ViewState(
        latitude=USC_CENTER_Y,
        longitude=USC_CENTER_X,
        zoom=12,
        pitch=0,
        bearing=0,
    )
    
    # Define map style
    map_style = "mapbox://styles/mapbox/light-v9"
    
    color_lookup = pdk.data_utils.assign_random_colors(subject_ids)
    
    # Load the GeoJSON file
    neighborhoods_data = './neighborhoods.geojson'

    # Create the GeoJsonLayer using the neighborhood data
    neighborhood_layer = pdk.Layer(
        'GeoJsonLayer',
        data=neighborhoods_data,
        opacity=0.5,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=False,
        get_line_color=[0, 255, 255],
        get_fill_color=[255, 0, 0],
        get_line_width=2,
        auto_highlight=True
    )
    
    # Add a button to go back to the input page
    if st.button("Back to Inputs"):
        # Go back to the input page
        session["page"] = "input"
        st.experimental_rerun()
        
    # creating a single-element container
    placeholder = st.empty()
    
    
    # near real-time / live feed simulation
    while True:
        if len(subject_ids) == 0:
            placeholder.info("Query resulted in no subjects! Select the subjects again.")
            break
        user_trajectories = {}
        df_hrate_full = st.session_state['df_hrate_full']
        df_calories_full = st.session_state['df_calories_full']
        df_coords_full = st.session_state['df_coords_full']
        new_hrates, new_calories, new_coords = get_data()  
        df_hrate_full = pd.concat([df_hrate_full, new_hrates])
        df_calories_full = pd.concat([df_calories_full, new_calories])
        df_coords_full = pd.concat([df_coords_full, new_coords])
        st.session_state['df_hrate_full'] = df_hrate_full
        st.session_state['df_calories_full'] = df_calories_full
        st.session_state['df_coords_full'] = df_coords_full
        df_mets_full = calculate_mets(df_calories_full)
        
        # filtering data
        df_hrate = df_hrate_full.loc[df_hrate_full['user_id'].isin(subject_ids)]
        df_calories = df_calories_full.loc[df_calories_full['user_id'].isin(subject_ids)]
        df_coords = df_coords_full.loc[df_coords_full['user_id'].isin(subject_ids)]
        df_mets = df_mets_full.loc[df_mets_full['user_id'].isin(subject_ids)]
        
        # creating KPIs
        avg_heart_rate = df_hrate['value'].mean()
        min_heart_rate = df_hrate['value'].min()
        max_heart_rate = df_hrate['value'].max()

        avg_calories = df_calories['value'].mean()
        min_calories = df_calories['value'].min()
        max_calories = df_calories['value'].max()
        avg_calories = df_calories['value'].mean()

        avg_mets = df_mets['value'].mean()
        min_mets = df_mets['value'].min()
        max_mets = df_mets['value'].max()
        avg_mets = df_mets['value'].mean()
        
        # getting window records
        window_end_time = df_hrate.index[-1]
        window_start_time = df_hrate.index[-1] - timedelta(seconds=window_size)
        
        window_hrate_df = df_hrate.loc[df_hrate.index >= window_start_time]
        window_calories_df = df_calories.loc[df_calories.index >= window_start_time]
        window_mets_df = df_mets.loc[df_mets.index >= window_start_time]
        
        avg_win_heart_rate = window_hrate_df['value'].mean()
        min_win_heart_rate = window_hrate_df['value'].min()
        max_win_heart_rate = window_hrate_df['value'].max()

        avg_win_calories = window_calories_df['value'].mean()
        min_win_calories = window_calories_df['value'].min()
        max_win_calories = window_calories_df['value'].max()
        avg_win_calories = window_calories_df['value'].mean()

        avg_win_mets = window_mets_df['value'].mean()
        min_win_mets = window_mets_df['value'].min()
        max_win_mets = window_mets_df['value'].max()
        avg_win_mets = window_mets_df['value'].mean()
        
        # get control group statistics
        control_stats = get_control_stats(df_hrate_full, df_calories_full, df_mets_full, control_ids=control_ids)
        win_control_stats = get_control_stats(df_hrate_full.loc[df_hrate_full.index>=window_start_time], 
                                        df_calories_full.loc[df_calories_full.index>=window_start_time], 
                                        df_mets_full.loc[df_mets_full.index>=window_start_time],
                                        control_ids=control_ids)
        
        # Add new data to user trajectories
        for user_id in df_coords["user_id"].unique():
            user_data = df_coords[df_coords["user_id"] == user_id][["lon", "lat"]]
            if user_id not in user_trajectories:
                user_trajectories[user_id] = {"coordinates": [user_data.values.tolist()], "width": 5}
            else:
                user_trajectories[user_id]["coordinates"][0] += (user_data.values.tolist())

        # Create Pydeck layers for each user's trajectory
        layers = [neighborhood_layer]
        for user_id, user_trajectory in user_trajectories.items():
            layer = create_layer(pd.DataFrame(user_trajectory), color=color_lookup[user_id])
            # layers.append(layer)
            layers += layer
            
        
        
        with placeholder.container():
            get_map_legend(color_lookup)
            # Update Pydeck map with new layers
            st.pydeck_chart(pdk.Deck(
                map_style=map_style,
                initial_view_state=view_state,
                layers=layers
            ))        
            
            
            st.markdown("#### Entire Selected Time")
            # create three columns
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

            # fill in those three columns with respective metrics or KPIs
            kpi1.metric(
                label="Average Heart-Rate",
                value=round(avg_heart_rate),
                delta=round(avg_heart_rate - control_stats['heart_rate']['avg']),
            )
            
            kpi2.metric(
                label="Min Heart-Rate",
                value=round(min_heart_rate, 2),
                delta=round(min_heart_rate - control_stats['heart_rate']['avg']),
            )
            
            kpi3.metric(
                label="Max Heart-Rate",
                value=round(max_heart_rate, 2),
                delta=round(max_heart_rate - control_stats['heart_rate']['avg']),
            )

            kpi4.metric(
                label='Avg Calories Burned',
                value=round(avg_calories, 2),
                delta=round(avg_calories - control_stats['calories']['avg'], 2),
            )

            kpi5.metric(
                label='Avg METs so far',
                value=round(avg_mets, 2),
                delta=round(avg_mets - control_stats['mets']['avg'], 2),
            )
            
            st.markdown("#### Selected Window")
            
            wkpi1, wkpi2, wkpi3, wkpi4, wkpi5 = st.columns(5)

            # fill in those three columns with respective metrics or KPIs
            wkpi1.metric(
                label="Average Window Heart-Rate",
                value=round(avg_win_heart_rate),
                delta=round(avg_win_heart_rate - control_stats['heart_rate']['avg']),
            )
            
            wkpi2.metric(
                label="Minimum Window Heart-Rate",
                value=round(min_win_heart_rate, 2),
                delta= round(min_win_heart_rate - control_stats['heart_rate']['avg']),
            )
            
            
            wkpi3.metric(
                label="Max Window Heart-Rate",
                value=round(max_win_heart_rate,2),
                delta=round(max_win_heart_rate - control_stats['heart_rate']['avg']),
            )

            wkpi4.metric(
                label="Avg Calories Burned in Last Window",
                value=round(avg_win_calories, 2),
                delta=round(avg_win_calories - control_stats['calories']['avg'], 2),
            )

            wkpi5.metric(
                label='Total METs in Last Window',
                value=round(avg_win_mets, 2),
                delta=round(avg_win_mets - control_stats['mets']['avg'], 2),
            )

            # create heart-rates chart
            fig_hrate = go.Figure()
            fig_calories = go.Figure()
            fig_mets = go.Figure()

            
            grouped_df_hrate = df_hrate.groupby('user_id')

            for user_id, group in grouped_df_hrate:
                fig_hrate.add_scattergl(x=group.index, y=group['value'],
                                        name=f'user_id: {user_id}')
            fig_hrate.update_layout(xaxis_title='Timestamp', yaxis_title='Value')
            add_aux_rectangles(fig_hrate, df_hrate, df_hrate_full, window_start_time, window_end_time)

            # plot calories for each user
            grouped_df_calories = df_calories.groupby('user_id')
            for user_id, group in grouped_df_calories:
                fig_calories.add_scattergl(x=group.index, y=group['value'], name=f'user_id: {user_id}')
            fig_calories.update_layout(xaxis_title='Timestamp', yaxis_title='Value')
            add_aux_rectangles(fig_calories, df_calories, df_calories_full, window_start_time, window_end_time)

            # plot mets for each user
            grouped_df_mets = df_mets.groupby('user_id')
            for user_id, group in grouped_df_mets:
                fig_mets.add_scattergl(x=group.index, y=group['value'], name=f'user_id: {user_id}')
            fig_mets.update_layout(xaxis_title='Timestamp', yaxis_title='Value')

            
            st.markdown("### Heart-Rate plot")
            # st.write(fig_hrate)
            st.plotly_chart(fig_hrate, use_container_width=True)
            st.markdown("### Calories plot")
            # st.write(fig_calories)
            st.plotly_chart(fig_calories, use_container_width=True)
            st.markdown("### METs plot")
            # st.write(fig_mets)
            st.plotly_chart(fig_mets, use_container_width=True)
            
            # st.line_chart(df['value'])
            # add barcharts to compare mean features to the global mean stats
            heart_rate_comp_data = {
                'Selected Subject(s) Average': [avg_heart_rate],
                'Control Group Average': [control_stats['heart_rate']['avg']]
            }

            calories_comp_data = {
                'Selected Subject(s) Average': [avg_calories],
                'Control Group Average': [control_stats['calories']['avg']]
            }

            mets_comp_data = {
                'Selected Subject(s) Average': [avg_mets],
                'Control Group Average': [control_stats['mets']['avg']]
            }

            # create a DataFrame from the dictionary
            df_heart_rate_comp = pd.DataFrame(heart_rate_comp_data)
            df_calories_comp = pd.DataFrame(calories_comp_data)
            df_mets_comp = pd.DataFrame(mets_comp_data)


            # add the title for the charts
            st.title("Comparison of Average Features for Selected Subject(s) to Control Group's Averages")
            # create three equally sized columns using st.beta_columns
            col1, col2, col3 = st.columns(3)
            # plot the first bar chart for heart rate in col1
            with col1:
                st.subheader('Heart Rate')
                # st.bar_chart(df_heart_rate_comp, width=150, height=300)
                fig_bar1 = get_bar_fig(df_heart_rate_comp, label='Heart Rate')
                # Display chart in Streamlit
                st.plotly_chart(fig_bar1, use_container_width=False)


            # plot the second bar chart for calories in col2
            with col2:
                st.subheader('Calories')
                fig_bar2 = get_bar_fig(df_calories_comp, label='Calories')
                # Display chart in Streamlit
                st.plotly_chart(fig_bar2, use_container_width=False)

            # plot the third bar chart for mets in col3
            with col3:
                st.subheader('METs')
                fig_bar3 = get_bar_fig(df_mets_comp, label='METs')
                # Display chart in Streamlit
                st.plotly_chart(fig_bar3, use_container_width=False)

            
            if not real_time_update:
                break
            time.sleep(TIMEOUT)




def main():
    # dashboard title
    st.title("Real-Time / Apple-Watch Heart-Rate Monitoring Dashboard")
    session = st.session_state
    
    # Display the appropriate page based on the session state
    if session is None:
        st.error("Please run the app first.")
        return

    if session.get("page", "input") == "input":
        garmin_df = get_garmin_df(get_db_engine())
        garmin_df.age = garmin_df.age.astype(int)
        garmin_df.weight = garmin_df.weight.astype(int)
        garmin_df.height = garmin_df.height.astype(int)
        input_page(garmin_df)
    elif session.get("page") == "results":
        results_page()


if __name__ == '__main__':
    main()