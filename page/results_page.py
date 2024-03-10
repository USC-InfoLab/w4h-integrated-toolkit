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
from page.import_page import import_page
import geopandas as gpd
from shapely import wkb
import os
import plotly.express as px

from lib.lib_utils import load_config,save_config,get_db_engine
from lib.lib_data_ingest import calculate_mets
from lib.lib_conf import *



# Flask server API endpoint
SERVER_URL = f"http://{HOST}:{PORT}"


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


def add_aux_rectangles(fig, df, df_full, window_start, window_end, real_time=False):
    if real_time:
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

    unsafe_values = df[(df['value'] < safe_min) | (df['value'] > safe_max)]
    # Set the number of windows and calculate the window size
    check_window_num = 600
    date_range = df.index[-1] - df.index[0]
    unsafe_check_window_size = max(date_range / check_window_num, timedelta(seconds=30))
    unsafe_check_window_start = df.index[0]
    if not unsafe_values.empty:
        while unsafe_check_window_start <= unsafe_values.index[-1]:
            num_unsafe_vals = unsafe_values[(unsafe_values.index >= unsafe_check_window_start) & (
                        unsafe_values.index < (unsafe_check_window_start + unsafe_check_window_size))].shape[0]
            num_all_vals = df[(df.index >= unsafe_check_window_start) & (
                        df.index < (unsafe_check_window_start + unsafe_check_window_size))].shape[0]
            if num_unsafe_vals > 0:
                fig.add_shape(
                    type='rect',
                    xref='x', yref='paper',
                    x0=unsafe_check_window_start, y0=0,
                    x1=unsafe_check_window_start + unsafe_check_window_size, y1=1,
                    fillcolor='red',
                    opacity=0.7 * (num_unsafe_vals / num_all_vals) + 0.2,
                    layer='below',
                    line_width=0
                )
            unsafe_check_window_start += unsafe_check_window_size

# Define function to create Pydeck layer
def create_layer(df, color):
    coordinates = df['coordinates']

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
        get_radius=150,
        get_fill_color=[0, 0, 255],
        pickable=True
    )

    marker_layer_end = pdk.Layer(
        "ScatterplotLayer",
        data=[{"position": coordinates.tolist()[-1]}],
        get_position="position",
        get_radius=150,
        get_fill_color=[255, 0, 0],
        pickable=True
    )

    return [layer, marker_layer_start, marker_layer_end]

# read data from Flask server (real-time) or from database (historical)
def get_data(session=None, real_time=False) -> pd.DataFrame:
    if real_time:
        response = requests.get(SERVER_URL, params={'db_name': st.session_state["current_db"]})
        data = response.json()
        # df_hrate = pd.DataFrame(data)
        df_hrate = pd.DataFrame(data['heart_rates'])
        df_calories = pd.DataFrame(data['calories'])
        df_coords = pd.DataFrame(data['coordinates'])
        df_coords['value'] = df_coords['value'].apply(lambda x: wkb.loads(bytes.fromhex(x)))
        df_coords = gpd.GeoDataFrame(df_coords, geometry='value')
        # df_hrate['timestamp'] = pd.to_datetime(df_hrate['timestamp'])
        # df_calories['timestamp'] = pd.to_datetime(df_calories['timestamp'])
        # df_coords['timestamp'] = pd.to_datetime(df_coords['timestamp'])
        # df_hrate = df_hrate.set_index('timestamp')
        # df_calories = df_calories.set_index('timestamp')
        # df_coords = df_coords.set_index('timestamp')
        # return df_hrate, df_calories, df_coords
    else:
        start_date = session.get('start_date')
        end_date = session.get('end_date')
        db_conn = get_db_engine(mixed_db_name=session["current_db"])
        # query heart rate
        df_hrate = pd.read_sql(
            f"SELECT * FROM {DB_TABLE} WHERE Date(timestamp) >= Date(%s) AND Date(timestamp) <= Date(%s)", db_conn,
            params=[start_date, end_date])
        df_hrate.sort_values(by=['timestamp'], inplace=True)
        # query calories
        df_calories = pd.read_sql(
            f"SELECT * FROM {DB_CALORIES_TABLE} WHERE Date(timestamp) >= Date(%s) AND Date(timestamp) <= Date(%s)",
            db_conn, params=[start_date, end_date])
        df_calories.sort_values(by=['timestamp'], inplace=True)
        # query coordinates
        df_coords = gpd.read_postgis(
            f"SELECT * FROM {DB_COORDINATES_TABLE} WHERE Date(timestamp) >= Date(%s) AND Date(timestamp) <= Date(%s)",
            db_conn, params=[start_date, end_date], geom_col='value')
        df_coords.sort_values(by=['timestamp'], inplace=True)

    df_hrate['timestamp'] = pd.to_datetime(df_hrate['timestamp'])
    df_hrate = df_hrate.set_index('timestamp')
    df_calories['timestamp'] = pd.to_datetime(df_calories['timestamp'])
    df_calories = df_calories.set_index('timestamp')
    df_coords['timestamp'] = pd.to_datetime(df_coords['timestamp'])
    df_coords = df_coords.set_index('timestamp')
    return df_hrate, df_calories, df_coords


def results_page(config):
    # Get the session state
    session = st.session_state
    if session is None:
        st.error("Please use the inputs page first.")
        return

    print('result page!')
    user_id_col_name = config['mapping']['columns']['user_id']
    subjects_df = session.get('subjects_df')
    subject_ids = subjects_df[user_id_col_name].tolist()
    control_df = session.get('control_df')
    control_ids = control_df[user_id_col_name].tolist()

    window_size = session['window_size']
    real_time_update = session['real_time_update']

    if real_time_update:
        # initialize the stream
        stream_start_date = session['stream_start_date']
        stream_start_time = session['stream_start_time']
        # send start datetime to the stream server
        stream_start_datetime = dt.combine(stream_start_date, stream_start_time)
        inited_start_datetime = requests.get(SERVER_URL + '/init_stream', params={'start_time': stream_start_datetime,
                                                                                  'db_name': st.session_state[
                                                                                      "current_db"]},
                                             verify=False).json()
        # restart dataframes
        st.session_state['df_hrate_full'] = pd.DataFrame()
        st.session_state['df_calories_full'] = pd.DataFrame()
        st.session_state['df_coords_full'] = gpd.GeoDataFrame()

    if 'df_hrate_full' not in st.session_state or 'df_calories_full' not in st.session_state or 'df_coords_full' not in st.session_state:
        st.session_state['df_hrate_full'] = pd.DataFrame()
        st.session_state['df_calories_full'] = pd.DataFrame()
        st.session_state['df_coords_full'] = gpd.GeoDataFrame()

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
        elif len(control_ids) == 0:
            placeholder.info("Query resulted in no control subjects! Select the control subjects again.")
        user_trajectories = {}
        df_hrate_full = st.session_state['df_hrate_full']
        df_calories_full = st.session_state['df_calories_full']
        df_coords_full = st.session_state['df_coords_full']
        new_hrates, new_calories, new_coords = get_data(session=session, real_time=real_time_update)
        df_hrate_full = pd.concat([df_hrate_full, new_hrates]) if real_time_update else new_hrates
        df_calories_full = pd.concat([df_calories_full, new_calories]) if real_time_update else new_calories
        df_coords_full = pd.concat([df_coords_full, new_coords]) if real_time_update else new_coords
        st.session_state['df_hrate_full'] = df_hrate_full
        st.session_state['df_calories_full'] = df_calories_full
        st.session_state['df_coords_full'] = df_coords_full
        df_mets_full = calculate_mets(df_calories_full)

        # filtering data
        # fix subject ids dtype
        user_id_dtype = df_hrate_full.user_id.dtype
        if user_id_dtype == np.int64:
            user_id_dtype = int
        # else if string
        elif user_id_dtype == object:
            user_id_dtype = str
        # cast subject ids and control ids to the same dtype as df_hrate dtype
        subject_ids = [user_id_dtype(item) for item in subject_ids]
        control_ids = [user_id_dtype(item) for item in control_ids]
        df_hrate = df_hrate_full.loc[df_hrate_full['user_id'].isin(subject_ids)]
        df_calories = df_calories_full.loc[df_calories_full['user_id'].isin(subject_ids)]
        df_coords = df_coords_full.loc[df_coords_full['user_id'].isin(subject_ids)]
        df_mets = df_mets_full.loc[df_mets_full['user_id'].isin(subject_ids)]
        df_email_date_range = df_mets.groupby('user_id')['datetime'].agg(start_date='min', end_date='max')
        df_email_date_range = df_email_date_range.reset_index().rename(columns={'user_id': 'user_id'})
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
        window_end_time = df_hrate.index[-1] if real_time_update and len(df_hrate) > 0 else pd.Timestamp(
            datetime.datetime.now(), tz='UTC')
        window_start_time = (df_hrate.index[-1] - timedelta(seconds=window_size)) if real_time_update and len(
            df_hrate) > 0 else pd.Timestamp(datetime.datetime.now(), tz='UTC')

        if real_time_update:
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
        if real_time_update:
            win_control_stats = get_control_stats(df_hrate_full.loc[df_hrate_full.index >= window_start_time],
                                                  df_calories_full.loc[df_calories_full.index >= window_start_time],
                                                  df_mets_full.loc[df_mets_full.index >= window_start_time],
                                                  control_ids=control_ids)

        # Add new data to user trajectories
        layers = [neighborhood_layer]
        for user_id in df_coords["user_id"].unique():
            user_data = df_coords[df_coords["user_id"] == user_id]
            df = pd.DataFrame(columns=['coordinates', 'width'])
            coordinate_dict = {"coordinates": [[y, x] for y, x in zip(user_data.value.y, user_data.value.x)],
                               "width": 5}
            df = df.append(coordinate_dict, ignore_index=True)
            layers += create_layer(df, color_lookup[user_id])
            # user_trajectories[user_id] = {"coordinates": [[y,x] for y,x in zip(user_data.value.y,user_data.value.x)], "width": 5}

        # Create Pydeck layers for each user's trajectory
        # for user_id, user_trajectory in user_trajectories.items():
        #     print(user_id)
        #     print(user_trajectory)
        #     print(pd.DataFrame(user_trajectory))
        #     layer = create_layer(pd.DataFrame(user_trajectory), color=color_lookup[user_id])
        #     # layers.append(layer)
        #     layers += layer

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
            try:
                kpi1.metric(
                    label="Average Heart-Rate",
                    value=round(avg_heart_rate),
                    delta=round(avg_heart_rate - control_stats['heart_rate']['avg']),
                )
            except Exception as e:
                traceback.print_exc()
                st.error(e)
                st.error("No data available for heart rate")
                break

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

            if real_time_update:
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
                    delta=round(min_win_heart_rate - control_stats['heart_rate']['avg']),
                )

                wkpi3.metric(
                    label="Max Window Heart-Rate",
                    value=round(max_win_heart_rate, 2),
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
            fig_aligned_mets = go.Figure()
            fig_project_dates = px.timeline(df_email_date_range, x_start="start_date", x_end="end_date", y="user_id",
                                            color="user_id")

            grouped_df_hrate = df_hrate.groupby('user_id')

            for user_id, group in grouped_df_hrate:
                fig_hrate.add_scatter(x=group.index, y=group['value'],
                                      name=f'user_id: {user_id}')
                fig_hrate.update_traces(showlegend=True)
            fig_hrate.update_layout(xaxis_title='Timestamp', yaxis_title='Value')
            add_aux_rectangles(fig_hrate, df_hrate, df_hrate_full, window_start_time, window_end_time,
                               real_time=real_time_update)

            # plot calories for each user
            grouped_df_calories = df_calories.groupby('user_id')
            for user_id, group in grouped_df_calories:
                group['datetime'] = pd.to_datetime(group.index)
                # group['value'] = np.where(group['datetime'].diff().shift(-1) > timedelta(hours=2), None, group['value'])
                fig_calories.add_bar(x=group.index, y=group['value'], name=f'user_id: {user_id}')
                # fig_calories.add_scatter(x=group.index, y=group['value'], name=f'user_id: {user_id}')
            fig_calories.update_layout(xaxis_title='Timestamp', yaxis_title='Value')
            # add_aux_rectangles(fig_calories, df_calories, df_calories_full, window_start_time, window_end_time, real_time=real_time_update)

            # plot mets for each user
            grouped_df_mets = df_mets.groupby('user_id')
            for user_id, group in grouped_df_mets:
                fig_mets.add_scatter(x=group.index, y=group['value'], name=f'user_id: {user_id}')
            fig_mets.update_layout(xaxis_title='Timestamp', yaxis_title='Value')

            # plot aligned mets for each user
            # print('df_mets_full: ',df_mets_full)
            # print('df_mets: ',df_mets)
            # st.write('df_mets_full')
            # st.write(df_mets_full)
            # st.write('df_mets')
            # st.write(df_mets)
            for user_id, group in grouped_df_mets:
                # st.write(user_id)
                # st.write(group)
                fig_aligned_mets.add_scatter(x=group.days_since_start, y=group['value'], name=f'user_id: {user_id}')
            fig_aligned_mets.update_layout(
                xaxis=dict(
                    rangeslider=dict(
                        visible=True
                    ),
                    tickformat=".2f",
                    title="Days (Decimal)",
                    # type="date"
                ),
                title='METS with available days',
                yaxis_title='Mets'
            )
            fig_project_dates.update_layout(
                xaxis_title='Time',
                yaxis_title='User Email',
                title='User Activity Duration',
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, label='1w', step='day', stepmode='backward'),
                            dict(count=1, label='1m', step='month', stepmode='backward'),
                            dict(count=6, label='6m', step='month', stepmode='backward'),
                            dict(step='all')
                        ])
                    ),
                    type='date'
                )
            )

            st.markdown("### Heart-Rate Plot")
            # st.write(fig_hrate)
            st.plotly_chart(fig_hrate, use_container_width=True)
            with st.expander("### Calories and METs Plots", expanded=False):
                st.markdown("#### Calories plot")
                # st.write(fig_calories)
                st.plotly_chart(fig_calories, use_container_width=True)
                st.markdown("#### METs plot")
                # st.write(fig_mets)
                st.plotly_chart(fig_mets, use_container_width=True)
                st.markdown("#### Aligned METs plot")
                # st.write(fig_aligned_mets)
                st.plotly_chart(fig_aligned_mets, use_container_width=True)

                st.markdown("#### Projected Start and End Dates")
                st.plotly_chart(fig_project_dates, use_container_width=True)

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

            if not real_time_update and session.get('num_time_ranges') > 0:
                # add the charts for selected ranges
                st.title("Analysis of Selected Time Ranges")
                num_time_ranges = session.get('num_time_ranges')
                time_ranges = session.get('time_ranges')
                time_ranges_labels = session.get('time_ranges_labels')
                for i in range(num_time_ranges):
                    range_start, range_end = time_ranges[i]
                    range_label = time_ranges_labels[i]
                    with st.expander(f'##### {range_label}: {range_start} to {range_end}', expanded=False):
                        # Get data for time range
                        time_range_hrate_df = df_hrate_full.loc[range_start:range_end]

                        # Filter data for subjects and control group
                        subjects_range_hrate_df = time_range_hrate_df.loc[
                            time_range_hrate_df['user_id'].isin(subject_ids)]
                        control_range_hrate_df = time_range_hrate_df.loc[
                            time_range_hrate_df['user_id'].isin(control_ids)]

                        # get stats for time range for each group
                        subjects_range_hrate_avg = subjects_range_hrate_df['value'].mean()
                        subjects_range_hrate_min = subjects_range_hrate_df['value'].min()
                        subjects_range_hrate_max = subjects_range_hrate_df['value'].max()

                        control_range_hrate_avg = control_range_hrate_df['value'].mean()
                        control_range_hrate_min = control_range_hrate_df['value'].min()
                        control_range_hrate_max = control_range_hrate_df['value'].max()

                        # visualize metrics in separate columns
                        # create three columns
                        kpi1, kpi2, kpi3 = st.columns(3)
                        kpi1.metric(
                            label=f"Average Heart-Rate in Time Range ({range_start} to {range_end})",
                            value=round(subjects_range_hrate_avg),
                            delta=round(subjects_range_hrate_avg - control_range_hrate_avg)
                        )

                        kpi2.metric(
                            label=f"Minimum Heart-Rate in Time Range ({range_start} to {range_end})",
                            value=round(subjects_range_hrate_min, 2),
                            delta=round(subjects_range_hrate_min - control_range_hrate_min, 2)
                        )

                        kpi3.metric(
                            label=f"Maximum Heart-Rate in Time Range ({range_start} to {range_end})",
                            value=round(subjects_range_hrate_max, 2),
                            delta=round(subjects_range_hrate_max - control_range_hrate_max, 2)
                        )

                        # visualize same metrics in bar charts
                        range_hrate_avg_comp = {
                            'Selected Subject(s) Average': [subjects_range_hrate_avg],
                            'Control Group Average': [control_range_hrate_avg]
                        }

                        range_hrate_min_comp = {
                            'Selected Subject(s) Minimum': [subjects_range_hrate_min],
                            'Control Group Minimum': [control_range_hrate_min]
                        }

                        range_hrate_max_comp = {
                            'Selected Subject(s) Maximum': [subjects_range_hrate_max],
                            'Control Group Maximum': [control_range_hrate_max]
                        }
                        # create corresponding dataframes
                        df_range_hrate_avg_comp = pd.DataFrame(range_hrate_avg_comp)
                        df_range_hrate_min_comp = pd.DataFrame(range_hrate_min_comp)
                        df_range_hrate_max_comp = pd.DataFrame(range_hrate_max_comp)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.subheader('Average Heart Rate')
                            fig_bar1 = get_bar_fig(df_range_hrate_avg_comp, label='Average Heart Rate')
                            # Display chart in Streamlit
                            st.plotly_chart(fig_bar1, use_container_width=False)
                        with col2:
                            st.subheader('Minimum Heart Rate')
                            fig_bar2 = get_bar_fig(df_range_hrate_min_comp, label='Minimum Heart Rate')
                            # Display chart in Streamlit
                            st.plotly_chart(fig_bar2, use_container_width=False)
                        with col3:
                            st.subheader('Maximum Heart Rate')
                            fig_bar3 = get_bar_fig(df_range_hrate_max_comp, label='Maximum Heart Rate')
                            # Display chart in Streamlit
                            st.plotly_chart(fig_bar3, use_container_width=False)

            # Show the dataframes and export to csv if needed
            st.title("Show/Export Data")
            with st.expander('Click to view more', expanded=False):
                st.header("Heart-Rate Data")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Full Heart Rate Data")
                    st.dataframe(df_hrate_full)
                    ste.download_button(
                        "Press to Download CSV",
                        df_hrate_full.to_csv().encode('utf-8'),
                        "df_hrate_full.csv",
                        "text/csv"
                    )
                with col2:
                    st.subheader("Selected Subject(s) Heart Rate Data")
                    st.dataframe(df_hrate)
                    ste.download_button(
                        "Press to Download CSV",
                        df_hrate.to_csv().encode('utf-8'),
                        "df_hrate_subjects.csv",
                        "text/csv"
                    )
                with col3:
                    df_hrate_control = df_hrate_full.loc[df_hrate_full['user_id'].isin(control_ids)]
                    st.subheader("Control Group Heart Rate Data")
                    st.dataframe(df_hrate_control)
                    ste.download_button(
                        "Press to Download CSV",
                        df_hrate_control.to_csv().encode('utf-8'),
                        "df_hrate_control.csv",
                        "text/csv"
                    )
        if not real_time_update:
            # reset dataframes
            st.session_state['df_hrate_full'] = pd.DataFrame()
            st.session_state['df_calories_full'] = pd.DataFrame()
            st.session_state['df_coords_full'] = pd.DataFrame()
            break
        time.sleep(session.get("timeout", TIMEOUT))