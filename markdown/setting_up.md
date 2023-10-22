# Welcome to w4h setting up toturial!

## Prerequisites
Ensure you have your database service in somewhere, and you know the 
db's host,username,password,database name, and the port it is listening to.

## Setup
1. create your config.yaml to setup your db, according to the example file:
[config.yaml.example](../app/static/config.yaml.example)
3. set up your DB server's config in the following field, and save file:
  - dbms: 'postgresql' or 'mysql'
  - host: your db's host
  - port: your db's port
  - user: your account's username
  - password: your account's password
3. upload your config file here: