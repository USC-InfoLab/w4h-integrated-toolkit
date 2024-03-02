import sqlite3

import yaml
import sqlalchemy
from sqlalchemy import create_engine
import urllib.parse

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
    db_engine = create_engine(con_str, echo=True)

    return db_engine

