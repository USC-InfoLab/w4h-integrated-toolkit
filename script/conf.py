# Sensor API
HOST = '127.0.0.1'
URL_PATH = 'fetchdata'
PORT = '9977'
DATA_TYPE = 'DATABASE' # 'DATABASE' or 'CSV'
DATASET = '<dataset_path>' # Path of the CSV dataset. Leave empty if you want to stream from database table

# Database Config

DB_TABLE = 'heart_rates'
DB_USER_TABLE = 'subjects'

# After each TIMEOUT (seconds), BATCH number of data entries is sent by the API
# Start time is the time recorded data stream is simulated to start from
DATE_TIME_COL = 'timestamp'
BATCH = 1
TIMEOUT = 5
START_TIME = '2022-01-01 8:20:00'
END_TIME = '2022-01-05 23:59:59'

# Slack Config
slack_token = 'xoxb-3720493889842-3733177258097-YImaXCHDSW35LvDQCDegfJ9L'
slack_channel = '#alerts'
slack_icon_emoji = ':health_worker:'
slack_user_name = 'W4H Alerts Bot'