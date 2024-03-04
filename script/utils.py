import sqlite3
import openai
import datetime

import yaml
import sqlalchemy
from sqlalchemy import create_engine
import urllib.parse
import json
import os

class Singleton(type):
    """Metaclass implementing the Singleton pattern.

    This metaclass ensures that only one instance of a class is created and shared among all instances.

    Attributes:
        _instances (dict): Dictionary holding the unique instances of each class.

    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        """Overrides the call behavior when creating an instance.

        This method checks if an instance of the class already exists. If not, it creates a new instance and
        stores it in the _instances dictionary.

        Args:
            cls (type): Class type.

        Returns:
            object: The instance of the class.

        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]




def load_config(config_file: str) -> dict:
    """Read the YAML config file

    Args:
        config_file (str): YAML configuration file path
    """
    with open(config_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
def save_config(config_file,config):
    with open(config_file, 'w') as file:
        yaml.dump(config, file)

def getServerIdByNickname(config_file: str='conf/config.yaml', nickname='local db'):
    config = load_config(config_file)
    server_number = config['database_number']
    for i in range(1,server_number+1):
        if(config["database"+str(i)]['nickname'] == nickname):
            return i
    raise Exception("No such nickname: \""+nickname+"\"")
def get_db_engine(config_file: str='conf/config.yaml',db_server_id = 1, db_server_nickname = None, db_name=None,mixed_db_name=None) -> sqlalchemy.engine.base.Engine:
    """Create a SQLAlchemy Engine instance based on the config file

    Args:
        config_file (str): Path to the config file
        db_name (str, optional): Name of the database to connect to. Defaults to None.

    Returns:
        sqlalchemy.engine.base.Engine: SQLAlchemy Engine instance for the database

    """
    # load the configurations
    config = load_config(config_file=config_file)
    # Database connection configuration
    if mixed_db_name != None:
        db_server_nickname = mixed_db_name.split("] ")[0][1:]
        db_name = mixed_db_name.split("] ")[1]
        print(mixed_db_name,"!")
        print("server: ", db_server_nickname,"!")
        print("db_name: ", db_name, "!")
    if db_server_nickname != None:
        db_server_id = getServerIdByNickname(nickname=db_server_nickname)
    db_server = 'database'+str(db_server_id)
    dbms = config[db_server]['dbms']
    db_host = config[db_server]['host']
    db_port = config[db_server]['port']
    db_user = config[db_server]['user']
    db_pass = config[db_server]['password']
    db_name = db_name if db_name else ''

    db_user_encoded = urllib.parse.quote_plus(db_user)
    db_pass_encoded = urllib.parse.quote_plus(db_pass)

    # creating SQLAlchemy Engine instance
    con_str = f'postgresql://{db_user_encoded}:{db_pass_encoded}@{db_host}:{db_port}/{db_name}'
    db_engine = create_engine(con_str, echo=True, future=True)

    return db_engine



def parse_query(query, default_values):
    openai.api_key = os.environ['OPENAI_API_KEY']  # Replace with your OpenAI API key

    prompt = f"""
    Parse the user's query and update the structured data for a data analysis application. Pay special attention to the age range, weight range, and height range specified by the user for both the subjects (the ones user wants to show) and the control group (the ones user wants to compare with). Extract these specific ranges from the query and apply them to the control group parameters in the JSON object. Retain the default values for any variables not explicitly mentioned in the user's query.
    default values:

    For Subjects:
    - selected_users: Default is {default_values['selected_users']}
    - selected_state_of_residence: Default is {default_values['selected_state_of_residence']}
    - selected_age_range: Default is {default_values['selected_age_range']}
    - selected_weight_range: Default is {default_values['selected_weight_range']}
    - selected_height_range: Default is {default_values['selected_height_range']}

    For Control Group:
    - selected_users_control: Default is {default_values['selected_users_control']}
    - selected_state_of_residence_control: Default is {default_values['selected_state_of_residence_control']}
    - selected_age_range_control: Default is {default_values['selected_age_range_control']}
    - selected_weight_range_control: Default is {default_values['selected_weight_range_control']}
    - selected_height_range_control: Default is {default_values['selected_height_range_control']}

    For Analysis Time Frame:
    - start_date: Default is {default_values['start_date']}
    - end_date: Default is {default_values['end_date']}

    User Query: "{query}"

    Based on the query, subject attributes are the ones user wants to show. Control group attributes are the ones the user wants to compare with. You should be able to find the wanted age, weight, and height ranges based on the input. provide the values for all previous variables in a JSON object. If a variable is not specified in the query, retain its default value. Please don't output any other text than the json object.
    The returned json object must have only the following keys: {default_values.keys()}
    """
    # prompt = f"""
    # Parse the following user query into a JSON object representing structured data for a data analysis application. The application has these input variables for subjects and control groups, along with analysis time frame and specific time ranges, with their default values:

    # For Subjects:
    # - selected_users: is list of subjects user wants to show.
    # - selected_state_of_residence: is a list of states of residence user wants to show. values should be from the list of states in the dataset: {default_values['selected_state_of_residence']}
    # - selected_age_range: is a list with two elements, the first one is the lower bound of the age range, and the second one is the upper bound.
    # - selected_weight_range: is a list with two elements, the first one is the lower bound of the weight range, and the second one is the upper bound.
    # - selected_height_range: is a list with two elements, the first one is the lower bound of the height range, and the second one is the upper bound.

    # For Control Group:
    # This is the attributes of the control group that the user wants to compare with the subject group.
    # - selected_users_control: is list of subjects user wants to show.
    # - selected_state_of_residence_control: is a list of states of residence user wants to show. values should be from the list of states in the dataset: {default_values['selected_state_of_residence_control']}
    # - selected_age_range_control: is a list with two elements, the first one is the lower bound of the age range, and the second one is the upper bound.
    # - selected_weight_range_control: is a list with two elements, the first one is the lower bound of the weight range, and the second one is the upper bound.
    # - selected_height_range_control: is a list with two elements, the first one is the lower bound of the height range, and the second one is the upper bound.

    # For Analysis Time Frame:
    # - start_date: Default is {default_values['start_date']}
    # - end_date: Default is {default_values['end_date']}

    # User Query: "{query}"

    # Based on the query, subject attributes are the ones user wants to show. If a variable is not specified in the query, return None for the variable.
    # The returned json object must have only the following keys: {default_values.keys()}
    # """

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ]
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error in querying OpenAI: {e}")
        return None

