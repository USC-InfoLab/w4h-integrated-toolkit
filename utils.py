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


def get_db_engine(config_file: str='conf/config.yaml', db_name=None) -> sqlalchemy.engine.base.Engine:
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
    dbms = config['database']['dbms']
    db_host = config['database']['host']
    db_port = config['database']['port']
    db_user = config['database']['user']
    db_pass = config['database']['password']
    db_name = db_name if db_name else ''

    db_user_encoded = urllib.parse.quote_plus(db_user)
    db_pass_encoded = urllib.parse.quote_plus(db_pass)

    # creating SQLAlchemy Engine instance
    con_str = f'postgresql://{db_user_encoded}:{db_pass_encoded}@{db_host}:{db_port}/{db_name}'
    db_engine = create_engine(con_str, echo=True, future=True)

    return db_engine

