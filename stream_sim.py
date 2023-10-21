import pandas as pd
from flask import Flask, jsonify, redirect, url_for, request
from sqlalchemy import create_engine
from loguru import logger
import urllib.parse

from script.conf import *
from script.utils import Singleton, load_config

data_loader_inited = False

app = Flask(__name__)
        

class DataLoader(metaclass=Singleton):
    """Singleton DataLoader class for loading and retrieving data in batches.

    This class provides functionality to load data and retrieve it in batches, maintaining a singleton
    instance throughout the application.

    Attributes:
        df (pandas.DataFrame): The underlying data in a pandas DataFrame.
        batch_size (int): The size of each data batch.
        row_ind (int): The current row index.

    """
    def __init__(self, dflist, batch_size=1, row_ind=0) -> None:
        """Initialize the DataLoader with data and parameters.

        Args:
            df (pandas.DataFrame): The underlying data in a pandas DataFrame.
            batch_size (int, optional): The size of each data batch. Defaults to 1.
            row_ind (int, optional): The initial row index. Defaults to 0.

        """
        self.dflist = dflist
        self.batch_size = batch_size
        self.row_ind = row_ind
        

        
    def get_next(self):
        """Get the next batch of data.

        This method retrieves the next batch of data from the underlying DataFrame based on the current
        row index and batch size. It also updates the row index for the next batch.

        Returns:
            pandas.DataFrame: The next batch of data.

        """
        full_res = {}
        features = ['heart_rates','calories','coordinates']
        for i, df in enumerate(self.dflist):
            target_times = (df.iloc[self.row_ind:].timestamp.unique())[:self.batch_size]
            res = df[df.timestamp.isin(target_times)]
            full_res[features[i]] = res
        # res['timestamp'] = res['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        self.row_ind += len(res)
        if self.row_ind >= len(self.dflist[0]):
            self.row_ind = 0
        return full_res
    
    
    def init_start_stream_index(self, start_time):
        """Initialize the row index to start streaming from.

        Args:
            start_time (str): The time to start streaming from.
        """
        if start_time is None:
            return self.dflist[0].iloc[self.row_ind].timestamp
        start_time = pd.Timestamp(start_time, tz='UTC')
        # remove timezone
        start_time = start_time.tz_localize(None)
        if start_time <= self.dflist[0].iloc[0].timestamp:
            self.row_ind = 0
        elif start_time >= self.dflist[0].iloc[-1].timestamp:
            self.row_ind = self.dflist[0].loc[self.dflist[0].timestamp == self.dflist[0].iloc[-1].timestamp].index.min()
        else:
            start_time_ind = self.dflist[0][self.dflist[0].timestamp < start_time].index.max() + 1
            self.row_ind = start_time_ind
        return self.dflist[0].iloc[self.row_ind].timestamp
        
            

def get_db_engine(db_name=None):
    """
    Returns a SQLAlchemy engine for connecting to the database.

    Returns:
        engine (sqlalchemy.engine.base.Engine): SQLAlchemy engine object.
    """
    config = load_config("conf/config.yaml")["database"]
    db_user_enc = urllib.parse.quote_plus(config["user"])
    db_pass_enc = urllib.parse.quote_plus(config["password"])
    # traceback.print_exc()
    return create_engine(f'{config["dbms"]}://{db_user_enc}:{db_pass_enc}@{config["host"]}:{config["port"]}/{db_name}')


def get_query_result(query, db_conn, params=[]):
    """
    Executes a SQL query on the database connection and returns the result as a Pandas DataFrame.

    Args:
        query (str): SQL query to be executed.
        db_conn (sqlalchemy.engine.base.Connection): Database connection object.
        params (list, optional): Parameters to be passed to the SQL query. Defaults to [].

    Returns:
        result (pandas.DataFrame): Result of the SQL query as a Pandas DataFrame.
    """
    return pd.read_sql(query, db_conn, params=params)



def get_series_from_db(db_conn, table_name, ids=None, id_column=None, start_time=None):
    """
    Retrieves the series data for specific IDs from the database starting from the specified time.

    Args:
        db_conn (sqlalchemy.engine.base.Connection): Database connection object.
        table_name (str): Name of the table to retrieve the series data from.
        ids (list or None, optional): IDs for which to retrieve the series data. Defaults to None.
        id_column (str or None, optional): Name of the ID column in the database table. Defaults to None.
        start_time (str or None, optional): Start time from which to retrieve the series data. Defaults to None.

    Returns:
        result (pandas.DataFrame): Result of the SQL query as a Pandas DataFrame.
    """
    conditions = []

    if ids is not None:
        conditions.append(f"{id_column} IN ({','.join(['%s'] * len(ids))})")

    if start_time is not None:
        conditions.append(f"timestamp >= '{start_time}'")

    condition_str = " AND ".join(conditions)

    if condition_str:
        condition_str = "WHERE " + condition_str

    query = f'''
    SELECT * FROM {table_name}
    {condition_str}
    ORDER BY timestamp ASC;
    '''

    params = ids if ids is not None else []

    return get_query_result(query, db_conn, params=params)


def get_data(config):
    """
    Retrieves the data from either a database table or a CSV file based on the configuration.

    Args:
        config (dict): Configuration dictionary containing the 'DATA_TYPE', 'DATASET' path, 'IDs', and 'START_TIME', DB_TABLE, DATE_TIME_COL.

    Returns:
        result (pandas.DataFrame): Data retrieved from the database table or CSV file.
    """
    data_type = config.get('DATA_TYPE', None)
    dataset_path = config.get('DATASET', None)
    ids = config.get('IDS', None)
    start_time = config.get('START_TIME', None)
    db_table = config.get('DB_TABLE', None)
    db_calories_table = config.get('DB_CALORIES_TABLE', None)
    db_coordinates_table = config.get('DB_COORDINATES_TABLE', None)
    date_time_col = config.get('DATE_TIME_COL', None)
    db_name = config.get('DB_NAME', None)
    
    if date_time_col is None:
        raise ValueError('DATE_TIME_COL must be specified.')
    if data_type != 'DATABASE' and data_type != 'CSV':
        raise ValueError('DATA_TYPE must be either DATABASE or CSV.')
    if data_type == 'DATABASE':
        # Fetching data from database table
        db_conn = get_db_engine(db_name)
        res = (get_series_from_db(db_conn, table_name=db_table, ids=ids, id_column='id', start_time=start_time),
               get_series_from_db(db_conn, table_name=db_calories_table, ids=ids, id_column='id', start_time=start_time),
               get_series_from_db(db_conn, table_name=db_coordinates_table, ids=ids, id_column='id', start_time=start_time))
        return res

    elif data_type == 'CSV':
        # Fetching data from CSV file
        df = pd.read_csv(dataset_path)
        df[f'{date_time_col}'] = pd.to_datetime(df[f'{date_time_col}'])
        # sort by timestamp
        df = df.sort_values(by=[f'{date_time_col}'])

        # Apply conditions if they exist
        if ids:
            df = df[df['id'].isin(ids)]
        if start_time:
            df = df[df[f'{date_time_col}'] >= start_time]

        return df 


def init_dataloader(inited=False,db_name=None,start_time=None):
    """
    Initializes the DataLoader with the fetched data based on the configuration.
    
    Args:
        inited (bool, optional): Whether the DataLoader has already been initialized. Defaults to False.

    Returns:
        DataLoader: Initialized DataLoader instance.
    """
    if not inited:
        config = {
            'DATA_TYPE': DATA_TYPE,
            'DATASET': DATASET,
            'IDS': get_ids(),
            'START_TIME': start_time,
            'DB_TABLE': DB_TABLE,
            'DB_CALORIES_TABLE': DB_CALORIES_TABLE,
            'DB_COORDINATES_TABLE': DB_COORDINATES_TABLE,
            'DATE_TIME_COL': DATE_TIME_COL,
            'DB_NAME': db_name
        }
        data = get_data(config)
        data_loader_inited = True
        return DataLoader(data, batch_size=BATCH)
    else:
        return DataLoader()


def get_ids():
    """
    Retrieves the target IDs.

    Returns:
        list or None: List of IDs or None if not specified in the config.
    """
    ids = None  # Default IDs if not specified in the config
    # You can implement your own logic to fetch IDs
    return ids


@app.route('/')
def root():
    """
    Redirects the root URL to the 'fetch_sensor_data' endpoint.

    Returns:
        Response: A redirection response to the 'fetch_sensor_data' endpoint.
    """
    db_name = request.args.get('db_name')
    return redirect(url_for('fetch_sensor_data',db_name=db_name))


@app.route('/init_stream', methods=['GET'])
def init_stream():
    """
    Initializes the start time from which to fetch the sensor data.

    Args:
        start_time: Start time from which to fetch the sensor data. Defaults to None.

    Returns:
        Response: A JSON response containing the initialized start time.
    """
    global data_loader_inited

    requested_start = request.args.get('start_time', None)
    db_name = request.args.get('db_name', None)
    data_loader = init_dataloader(inited=data_loader_inited,db_name=db_name,start_time = requested_start)
    inited_start_time = data_loader.init_start_stream_index(requested_start)
    resp = jsonify({'start_time': inited_start_time})
    logger.debug(resp)
    return resp


@app.route(f'/{URL_PATH}')
def fetch_sensor_data():
    """
    Fetches sensor data from the data loader and returns it as a JSON response.

    Returns:
        Response: A JSON response containing the fetched sensor data.
    """
    global data_loader_inited

    db_name = request.args.get('db_name')
    data_loader = init_dataloader(inited=data_loader_inited,db_name=db_name)
    data_loader_inited = True
    df_dict = data_loader.get_next()
    print(df_dict)
    res = {}
    for key, rows in df_dict.items():
        rows_with_strftime = rows.copy()
        rows_dict = rows_with_strftime.to_dict('records')
        res[key] = rows_dict
    resp = jsonify(res)
    logger.debug(resp)
    return resp


if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=False)

