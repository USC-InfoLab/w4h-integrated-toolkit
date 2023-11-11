# Welcome to w4h setting up toturial!

## Prerequisites
W4h contains a default postgre db which has been setting up in you server, uses port 5432.  
If you want to add a new or change the current db server, ensure you have your database service in somewhere, 
and you know the db's host,username,password,database name, and the port it is listening to.

## Setup
1. Change your config.yaml to setup your db, according to the example file:
[config.yaml.example](../app/static/config.yaml.example) 
2. Set the "database_number", Create new database server fields and set the index correctly.
3. set up your DB server's config in the following field, and save file:
  - nickname: your db's showing name in w4h
  - dbms: 'postgresql' or 'mysql'
  - host: your db's host
  - port: your db's port
  - user: your account's username
  - password: your account's password
3. upload your config file here: