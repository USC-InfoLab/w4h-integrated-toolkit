database_number: 1

database1:
  nickname: 'local db' # will shows in the selectbox
  dbms: 'postgresql'
  host: 'db' # Replace with your DB host
  port: 5432 # Replace with your DB port
  user: 'admin' # Replace with your DB username
  password: 'admin'

#database2:
#  nickname: <your nick name>
#  dbms: <db's system>
#  host: <db host>
#  port: <db port>
#  user: <db username>
#  password: <password>



mapping:
  columns:
    user_id: 'user_id'
    timestamp: 'timestamp'
    value: 'value'
  tables:
    user_table:
      name: 'geomts_users'
      columns:
        user_id: String(50)
        device: String(50)
        location: String(50)
    time_series:
      - heart_rates
      - calories
      - distances
      - steps
      - sleep
      - weight
    geo:
      - locations
