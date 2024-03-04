import datetime
import os
import sqlite3
import json
import pickle
import streamlit as st
from loguru import logger
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, ForeignKey, DateTime, REAL, Integer, Float, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database


from geoalchemy2 import Geometry

from script.utils import load_config, get_db_engine


def create_tables(db_server_nickname:str, db_name: str, config_file='conf/config.yaml'):
    """Create the W4H tables in the database with the given name based on the config file

    Args:
        db_name (str): Name of the database to create the tables in
        config_file (str, optional): Path to the config file. Defaults to 'conf/config.yaml'.
    """
    metadata = MetaData()
    config = load_config(config_file=config_file)
    db_engine = get_db_engine(db_server_nickname=db_server_nickname, db_name=db_name)
    # try:
    columns_config = config["mapping"]["columns"]

    # Create the user table
    user_table_config = config["mapping"]["tables"]["user_table"]
    dtype_mappings = config['mapping']['data_type_mappings']
    user_columns = [eval(f'Column("{col_attribute["name"]}", {dtype_mappings[col_attribute["type"]]}, primary_key={col_attribute["name"] == columns_config["user_id"]})') for col_attribute in user_table_config["attributes"]]  # Convert string to actual SQLAlchemy type
    user_table = Table(user_table_config["name"], metadata, *user_columns)


    # Create time series tables
    for table_name in config["mapping"]["tables"]["time_series"]:
        table = Table(table_name, metadata,
            Column(columns_config["user_id"], ForeignKey(user_table_config["name"] + '.' + columns_config["user_id"]), primary_key=True),
            Column(columns_config["timestamp"], DateTime, primary_key=True),
            Column(columns_config["value"], REAL),
        )

    # Create geo tables
    for table_name in config["mapping"]["tables"]["geo"]:
        table = Table(table_name, metadata,
            Column(columns_config["user_id"], ForeignKey(user_table_config["name"] + '.' + columns_config["user_id"]), primary_key=True),
            Column(columns_config["timestamp"], DateTime, primary_key=True),
            Column(columns_config["value"], Geometry('POINT'))
        )

    metadata.create_all(db_engine)
    # except Exception as err:
    #     db_engine.dispose()
    #     logger.error(err)
    
        
        
def create_w4h_instance(db_server:str, db_name: str, config_file='conf/config.yaml'):
    """Create a new W4H database instance with the given name and initialize the tables based on the config file

    Args:
        db_name (str): Name of the database to create
        config_file (str, optional): Path to the config file. Defaults to 'conf/config.yaml'.
    """
    db_engine_tmp = get_db_engine(db_server_nickname=db_server)
    try:
        logger.info('Database engine created!')
        # Execute the SQL command to create the database if it doesn't exist
        if not database_exists(f'{db_engine_tmp.url}{db_name}'):
            create_database(f'{db_engine_tmp.url}{db_name}')
            logger.success(f"Database {db_name} created!")
            db_engine_tmp.dispose()
        else:
            logger.error(f"Database {db_name} already exists!")
            db_engine_tmp.dispose()
            return
    except Exception as err:
        logger.error(err)
        db_engine_tmp.dispose()
    db_engine = get_db_engine(db_server_nickname=db_server, db_name=db_name)
    try:
        # Enable PostGIS extension
        with db_engine.connect() as connection:
            connection.execute(text(f"CREATE EXTENSION postgis;"))
            logger.success(f"PostGIS extension enabled for {db_name}!")
        db_engine.dispose()
    except Exception as err:
        logger.error(err)
        db_engine.dispose()
        return
    # Create the W4H tables
    create_tables(config_file=config_file, db_name=db_name, db_server_nickname=db_server)
    logger.success(f"W4H tables initialized!")
    
    
def get_existing_databases(config_file='conf/db_config.yaml') -> list:
    """Get a list of all existing databases

    Args:
        config_file (str, optional): Path to the config file. Defaults to 'conf/config.yaml'.

    Returns:
        list: List of all existing databases (strings)
    """
    db_list = []
    config = load_config(config_file=config_file)
    database_number = config['database_number']
    for i in range(1,database_number+1):
        db_engine = get_db_engine(db_server_id=i)
        try:
            with db_engine.connect() as connection:
                result = connection.execute(text("SELECT datname FROM pg_database WHERE datistemplate = false;"))
                db_list += [ '[' + config['database'+str(i)]['nickname'] + '] ' + row[0] for row in result]
            db_engine.dispose()
        except Exception as err:
            logger.error(err)
            db_engine.dispose()
            return db_list
    return db_list

def get_existing_database_server(config_file='conf/db_config.yaml') -> list:
    db_list_server = []
    config = load_config(config_file=config_file)
    database_number = config['database_number']
    for i in range(1, database_number + 1):
        db_list_server += [config['database'+str(i)]['nickname'] + ' (' + config['database'+str(i)]['host'] + ')']
    return db_list_server




def populate_tables(df: pd.DataFrame, db_name: str, mappings: dict, config_path='conf/config.yaml'):
    """Populate the W4H tables in the given database with the data from the given dataframe based on 
    the mappings between the CSV columns and the database tables.

    Args:
        df (pd.DataFrame): Dataframe containing the data to be inserted into the database
        db_name (str): Name of the database to insert the data into
        mappings (dict): Dictionary containing the mappings between the CSV columns and the database tables
        config_path (str, optional): Path to the config file. Defaults to 'conf/config.yaml'.
    """
    # Load the config
    config = load_config(config_path)
    
    # Extract default column names from the config
    default_user_id = config['mapping']['columns']['user_id']
    default_timestamp = config['mapping']['columns']['timestamp']
    default_value = config['mapping']['columns']['value']
    user_table_name = config['mapping']['tables']['user_table']['name']

    # Create a session
    engine = get_db_engine(mixed_db_name=db_name)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Ensure all unique users from the dataframe exist in the user table
    unique_users = df[mappings[default_user_id]].unique().astype(str)
    existing_users = session.query(Table(user_table_name, MetaData(bind=engine), autoload=True).c[default_user_id]).all()
    existing_users = [x[0] for x in existing_users]

    # Identify users that are not yet in the database
    new_users = set(unique_users) - set(existing_users)

    if new_users:
        # Convert the set of new users into a DataFrame
        all_new_users = pd.DataFrame({default_user_id: list(new_users)})
        
        # Use to_sql to insert all new users into the user table
        all_new_users.to_sql(user_table_name, engine, if_exists='append', index=False)

    # Get the subset of mappings that doesn't include default_user_id and default_timestamp
    table_mappings = {k: v for k, v in mappings.items() if k not in [default_user_id, default_timestamp]}
    
    # Loop through each table in table_mappings
    for table_name, csv_column in table_mappings.items():
        # Check if the mapping is not NULL and exists in the df
        if csv_column and csv_column in df.columns:
            
            # Ensure that the dataframe columns match the user_id, timestamp, and value from your CSV
            columns_to_insert = [mappings[default_user_id], mappings[default_timestamp], csv_column]
            
            subset_df = df[columns_to_insert].copy()
            
            # Rename columns to match the table's column names using the defaults from config
            subset_df.columns = [default_user_id, default_timestamp, default_value]
            
            # dropping duplicate user_id and timestamp
            subset_df.drop_duplicates(subset=[default_user_id, default_timestamp], inplace=True)
            # subset_df = subset_df.groupby([default_user_id, default_timestamp]).mean().reset_index()
            
            # handling geometry data
            if table_name in config["mapping"]["tables"]["geo"]:
                subset_df[default_value] = subset_df[default_value].apply(lambda x: f'POINT{x}'.replace(',', ''))
            
            # Insert data into the table
            subset_df.to_sql(table_name, engine, if_exists='append', index=False)

    # Commit the remaining changes and close the session
    session.commit()
    session.close()
    engine.dispose()


def populate_subject_table(df: pd.DataFrame, db_name: str, mappings: dict, config_path='conf/config.yaml'):
    """Populate the W4H tables in the given database with the data from the given dataframe based on
    the mappings between the CSV columns and the database tables.

    Args:
        df (pd.DataFrame): Dataframe containing the data to be inserted into the database
        db_name (str): Name of the database to insert the data into
        mappings (dict): Dictionary containing the mappings between the CSV columns and the database tables
        config_path (str, optional): Path to the config file. Defaults to 'conf/config.yaml'.
    """
    # Load the config
    config = load_config(config_path)

    # Create a session
    engine = get_db_engine(mixed_db_name=db_name)

    # create a user table dataframe using the mappings
    user_tbl_name = config['mapping']['tables']['user_table']['name']
    user_df = pd.DataFrame()
    for k, v in mappings.items():
        if v is not None:
            user_df[k] = df[v]
    # populate the user table (directly push df to table), if already exists, append new users
    # if columns don't exist, ignore
    user_df.to_sql(user_tbl_name, engine, if_exists='append', index=False)

    # Commit the remaining changes and close the session
    engine.dispose()



def getCurrentDbByUsername(username):
    with sqlite3.connect('user.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''select current_db from users where username = ?''',(username,))
        result = cursor.fetchone()
    return result[0]

def updateCurrentDbByUsername(username,currentDb):
    with sqlite3.connect('user.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''update users set current_db = ? where username = ?''',(currentDb,username,))
        conn.commit()

def saveSessionByUsername(session):
    with sqlite3.connect('user.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''select query_history from users where username = ?''',(session.data.get('login-username'),))
        result = cursor.fetchone()
        conn.commit()
    query_history = pickle.loads(result[0])
    # print("history:",query_history[0].get('selected_users'))
    query_history.append(session)
    serialized_object = pickle.dumps(query_history)

    with sqlite3.connect('user.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''UPDATE users SET query_history = ? WHERE username = ?''', (serialized_object,session.data['login-username'],))
        conn.commit()

def getSessionByUsername(username):
    with sqlite3.connect('user.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''select query_history from users where username = ?''',(username,))
        result = cursor.fetchone()
        conn.commit()

    return pickle.loads(result[0])