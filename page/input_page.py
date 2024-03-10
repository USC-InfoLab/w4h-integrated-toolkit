import streamlit as st
from lib.lib_utils import *
from lib.lib_conf import *
from lib.lib_data_ingest import *
from datetime import time as dt_time



# get full user info
def get_users_df(db_conn, config, pattern=None):
    query = f"SELECT * FROM {config['mapping']['tables']['user_table']['name']}"
    params = []
    if pattern:
        query += f" WHERE {config['mapping']['columns']['user_id']} LIKE %s"
        pattern = f'%{pattern}%'
        params = [pattern]
    # execute the query
    return pd.read_sql(query, db_conn, params=params)

# Function to create a widget based on attribute type and store the input
def create_widget(attribute, default_values, tag='subject', session=st.session_state):
    name = attribute['name']
    attr_type = attribute['type']
    description = attribute['description']
    label = f'{name}: {description}'
    key = f'{tag}_{name}'
    default_val = default_values[name]
    val = session.get(f'selected_{key}', default_val)
    if attr_type == 'int':
        min_val = default_val[0]
        max_val = default_val[1]
        input_item = st.slider(label, min_value=min_val, max_value=max_val, key=key, step=1, value=val)
    elif attr_type == 'float':
        min_val = default_val[0]
        max_val = default_val[1]
        input_item = st.slider(label, min_value=min_val, max_value=max_val, key=key, step=0.1, value=val)
    elif attr_type == 'string':
        possible_values = default_val
        input_item = st.multiselect(f'{label} (Blank allows all)', options=possible_values, key=key, default=val)
        if len(input_item) == 0:
            input_item = possible_values
    elif attr_type == 'datetime':
        min_val = default_val[0]
        max_val = default_val[1]
        input_item = st.date_input(label, min_value=min_val, max_value=max_val, key=key, value=val)
    elif attr_type == 'boolean':
        input_item = st.checkbox(label, key=key, value=val)
    else:
        raise ValueError(f"Attribute type {attr_type} not supported.")
    return input_item

# Function to create a widget based on attribute type and store the input
def create_default_values(attributes, db_conn, config):
    default_values = dict()
    db_user_table = config['mapping']['tables']['user_table']['name']
    for attribute in attributes:
        name = attribute['name']
        attr_type = attribute['type']
        if name == config['mapping']['columns']['user_id']:
            continue
        if attr_type == 'int':
            min_val, max_val = pd.read_sql(f"SELECT min({name}), max({name}) FROM {db_user_table}", db_conn).values.squeeze()
            min_val = int(min_val)
            max_val = int(max_val)
            default_values[name] = (min_val, max_val)
        elif attr_type == 'float':
            min_val, max_val = pd.read_sql(f"SELECT min({name}), max({name}) FROM {db_user_table}", db_conn).values.squeeze()
            min_val = float(min_val)
            max_val = float(max_val)
            default_values[name] = (min_val, max_val)
        elif attr_type == 'string':
            possible_values = pd.read_sql(f"SELECT distinct({name}) FROM {db_user_table}", db_conn).values.squeeze().tolist()
            default_values[name] = possible_values
        elif attr_type == 'datetime':
            min_val, max_val = pd.read_sql(f"SELECT min({name}), max({name}) FROM {db_user_table}", db_conn).values.squeeze()
            min_val = pd.to_datetime(min_val)
            max_val = pd.to_datetime(max_val)
            default_values[name] = (min_val, max_val)
        elif attr_type == 'boolean':
            default_values[name] = False
        else:
            raise ValueError(f"Attribute type {attr_type} not supported.")
    return default_values


def filter_users(df, attributes, ignore_nulls=True):
    for attribute in attributes.values():
        name = attribute['name']
        attr_type = attribute['type']
        ignore_nulls_str = f'or {name}.isnull()' if ignore_nulls else ''
        if attr_type == 'int':
            df = df.query(f"({name} >= {attribute['value'][0]} and {name} <= {attribute['value'][1]}) {ignore_nulls_str}")
        elif attr_type == 'float':
            df = df.query(f"{name} >= {attribute['value'][0]} and {name} <= {attribute['value'][1]} {ignore_nulls_str}")
        elif attr_type == 'string':
            df = df.query(f"{name} in {attribute['value']} {ignore_nulls_str}")
        elif attr_type == 'datetime':
            df = df.query(f"{name} >= '{attribute['value'][0]}' and {name} <= '{attribute['value'][1]}' {ignore_nulls_str}")
        elif attr_type == 'boolean':
            df = df.query(f"{name} == {attribute['value']} {ignore_nulls_str}")
        else:
            raise ValueError(f"Attribute type {attr_type} not supported.")
    return df

def create_filter_dict(attributes, config, selected_attrs):
    filter_dict = dict()
    for attribute in attributes:
        if attribute['name'] == config['mapping']['columns']['user_id']:
            continue
        item = attribute.copy()
        item['value'] = selected_attrs[attribute['name']]
        filter_dict[attribute['name']] = item
    return filter_dict

def input_page(config):
    global TIMEOUT
    # Get the session state
    session = st.session_state
    if session is None:
        st.error("Please run the app first.")
        return

    # get the user table config
    user_config = config['mapping']['tables']['user_table']

    # Connect to the database
    db_conn = get_db_engine(mixed_db_name=session["current_db"])
    # get the list of user id's
    user_ids = pd.read_sql(
        f"SELECT distinct({config['mapping']['columns']['user_id']}) FROM {config['mapping']['tables']['user_table']['name']}",
        db_conn).values.squeeze().tolist()

    # top-level filters

    # Selecting the Subjects
    st.header("Select Subject(s)")
    # add selector for user
    subject_selection_options = ['id', 'attribute']
    subject_selection_type = st.radio("Select subject(s) by id or by attribute?", subject_selection_options,
                                      index=session.get('subject_selection_type', 0))

    selected_users = []
    if subject_selection_type == 'id':
        selected_users = st.multiselect(
            "Select Subject ID(s) (Blank allows all)",
            options=user_ids,
            default=session.get('selected_users', []))
        if len(selected_users) > 0:
            temp_select_users = selected_users
        else:
            temp_select_users = user_ids

    selected_subj_attributes = dict()

    attrs_size_per_row = config['display_options']['input_page']['attributes_per_row_size']
    default_attr_values = create_default_values(user_config['attributes'], db_conn, config)
    if subject_selection_type == 'attribute':
        st.subheader("Select Subject(s) Attributes")
        counter = 0
        for attribute in user_config['attributes']:
            if counter % len(attrs_size_per_row) == 0:
                cols = st.columns(spec=attrs_size_per_row, gap='large')
            with cols[counter % len(attrs_size_per_row)]:
                if attribute['name'] == config['mapping']['columns']['user_id']:
                    continue
                selected_subj_attributes[attribute['name']] = create_widget(attribute, default_attr_values,
                                                                            tag='subject')
                counter += 1

    # Selecting the control group
    st.header("Select Control Group")
    # add selector for user
    control_selection_options = ['all', 'id', 'attribute']
    control_selection_type = st.radio(
        "Select control group (either as all studied individuals or filter by id or attribute)?",
        control_selection_options,
        index=session.get('control_selection_type', 0))

    selected_users_control = []
    if control_selection_type == 'id':
        selected_users_control = st.multiselect(
            "Select Control Target ID(s) (Blank allows all)",
            options=user_ids,
            default=session.get('selected_users_control', [])
        )
        if len(selected_users_control) > 0:
            temp_select_users_control = selected_users_control
        else:
            temp_select_users_control = user_ids

    selected_control_attributes = dict()
    if control_selection_type == 'attribute':
        st.subheader("Select Control Group Attributes")
        counter = 0
        for attribute in user_config['attributes']:
            if counter % len(attrs_size_per_row) == 0:
                cols = st.columns(spec=attrs_size_per_row, gap='large')
            with cols[counter % len(attrs_size_per_row)]:
                if attribute['name'] == config['mapping']['columns']['user_id']:
                    continue
                selected_control_attributes[attribute['name']] = create_widget(attribute, default_attr_values,
                                                                               tag='control')
                counter += 1

    st.header("Visualization/Analysis Configuration")
    real_time_update = st.checkbox("Real-Time stream simulation?", value=session.get("real_time_update", False))

    if not real_time_update:
        start_date = st.date_input(
            "Start date",
            session.get("start_date", datetime.datetime.strptime(START_TIME, '%Y-%m-%d %H:%M:%S'))
        )

        end_date = st.date_input(
            "End date",
            session.get("end_date", datetime.datetime.strptime(END_TIME, '%Y-%m-%d %H:%M:%S'))
        )

        st.markdown("#### Need to analyze specific time range? Select how many range(s) you want to analyze.")
        num_time_ranges = st.selectbox("Select how many time range(s) you want to analyze", range(0, 10),
                                       index=session.get('num_time_ranges', 3))
        def_time_ranges = [
            (dt_time(6, 45), dt_time(9, 30)),
            (dt_time(12, 30), dt_time(16, 0)),
            (dt_time(20, 0), dt_time(4, 45))
        ]
        def_time_ranges_labels = ['Workout #1', 'Workout #2', 'Sleep Schedule']
        time_ranges = session.get('time_ranges', def_time_ranges)
        time_ranges_labels = session.get('time_ranges_labels', def_time_ranges_labels)
        if num_time_ranges > 0:
            with st.expander(f"###### Time Ranges"):
                updated_ranges = []
                updated_range_labels = []
                for i in range(num_time_ranges):
                    # 2 columns for each time range
                    col1, col2, col3 = st.columns(spec=[1, 2, 2])
                    with col1:
                        range_label = st.text_input(f"Label for range {i + 1}", value=(
                            time_ranges_labels[i] if i < len(time_ranges_labels) else f"Time range {i + 1}"))
                    with col2:
                        range_start = st.time_input(f"Start time for range {i + 1}", value=(
                            time_ranges[i][0] if i < len(time_ranges) else dt_time(0, 0)))
                    with col3:
                        range_end = st.time_input(f"End time for range {i + 1}",
                                                  value=(time_ranges[i][1] if i < len(time_ranges) else dt_time(0, 0)))
                    updated_ranges.append((range_start, range_end))
                    updated_range_labels.append(range_label)
                    # st.divider()
                time_ranges = updated_ranges
                time_ranges_labels = updated_range_labels
    else:
        col1, col2 = st.columns(2)
        with col1:
            stream_start_date = st.date_input(
                "Start Date for Simulating Real-Time Stream",
                session.get("stream_start_date", datetime.datetime.strptime(START_TIME, '%Y-%m-%d %H:%M:%S'))
            )
        with col2:
            stream_start_time = st.time_input(
                "Start Time for Simulating Real-Time Stream",
                session.get("stream_start_time", datetime.datetime.strptime(START_TIME, '%Y-%m-%d %H:%M:%S'))
            )

    if real_time_update:
        window_size = st.number_input('Window Size (seconds)', value=session.get("window_size", DEFAULT_WINDOW_SIZE),
                                      step=15)
        TIMEOUT = st.number_input('Fast Forward (Every 1 Hour Equals How Many Seconds?)',
                                  value=session.get('timeout', float(TIMEOUT)), step=float(1), format="%.1f",
                                  min_value=0.1, max_value=float(100))

    # Add a button to go to the results page
    if st.button("Show Results"):

        # save input values to the session state
        session['real_time_update'] = real_time_update
        if not real_time_update:
            session['start_date'] = start_date
            session['end_date'] = end_date
            session['num_time_ranges'] = num_time_ranges
            session['time_ranges'] = time_ranges
            session['time_ranges_labels'] = time_ranges_labels
        elif real_time_update:
            session['stream_start_date'] = stream_start_date
            session['stream_start_time'] = stream_start_time
            session['timeout'] = TIMEOUT
        session["window_size"] = window_size if real_time_update else DEFAULT_WINDOW_SIZE
        session["real_time_update"] = real_time_update
        session['subject_selection_type'] = 0 if subject_selection_type == 'id' else 1
        session[
            'control_selection_type'] = 0 if control_selection_type == 'all' else 1 if control_selection_type == 'id' else 2

        session['selected_subj_attributes'] = selected_subj_attributes
        session['selected_control_attributes'] = selected_control_attributes

        session['selected_users'] = selected_users if subject_selection_type == 'id' else []
        session['selected_users_control'] = selected_users_control if control_selection_type == 'id' else []

        for name, value in selected_subj_attributes.items():
            session[f'selected_subject_{name}'] = value
        for name, value in selected_control_attributes.items():
            session[f'selected_control_{name}'] = value

        # get full user table
        user_df = get_users_df(db_conn, config)
        user_id_col_name = config['mapping']['columns']['user_id']
        # Filter the dataframe based on the selected criteria for subjects
        if subject_selection_type == 'id':
            subjects_df = user_df.query(f'{user_id_col_name} in @temp_select_users')
        else:
            subjects_filter = create_filter_dict(user_config['attributes'], config, selected_subj_attributes)
            subjects_df = filter_users(user_df, subjects_filter)

        # Filter the dataframe based on the selected criteria for control group
        if control_selection_type == 'all':
            control_df = user_df
        elif control_selection_type == 'id':
            control_df = user_df.query(f'{user_id_col_name} in @temp_select_users_control')
        else:
            control_filter = create_filter_dict(user_config['attributes'], config, selected_control_attributes)
            control_df = filter_users(user_df, control_filter)

        # Store the filtered dataframe in session state
        session['subjects_df'] = subjects_df
        session['control_df'] = control_df

        q = query_history(session)
        # print('q:qqqq: ',q)
        getSessionByUsername(q.data['login-username'])
        saveSessionByUsername(q)

        # Go to the results page
        session['page'] = "results"

        st.experimental_rerun()