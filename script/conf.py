# Sensor API
HOST = '127.0.0.1'
URL_PATH = 'fetchdata'
PORT = '9977'
DATA_TYPE = 'DATABASE' # 'DATABASE' or 'CSV'
DATASET = '<dataset_path>' # Path of the CSV dataset. Leave empty if you want to stream from database table

# Database Config

DB_TABLE = 'heart_rates'
DB_USER_TABLE = 'subjects'
DB_CALORIES_TABLE = 'calories'
DB_COORDINATES_TABLE = 'locations'

# After each TIMEOUT (seconds), BATCH number of data entries is sent by the API
# Start time is the time recorded data stream is simulated to start from
DATE_TIME_COL = 'timestamp'
BATCH = 1
TIMEOUT = 5
START_TIME = '2016-08-01 12:00:00'
END_TIME = '2016-08-14 23:59:59'