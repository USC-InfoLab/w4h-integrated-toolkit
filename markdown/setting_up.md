# Welcome to w4h setting up toturial!

## Prerequisites
Ensure you have your database service in somewhere, and you know the 
db's host,username,password,database name, and the port it is listening to.

## Setup
1. create your config.yaml to setup your db, according to the example file:
[config.yaml.example](../app/static/config.yaml.example)
3. create a file named config.yaml, then copy code in the file you downloaded(config.yaml.example), replace these fields:  
- database:
  - dbms: 'postgresql'
  - host: your db's host
  - port: your db's port
  - user: your db's username
  - password: your db's password
3. rename the file to config.yaml and put it into a directory named "conf"
4. shutdown the current docker container
5. rerun the docker:  
  ```shell
  docker run -dp 8501:8501 -v {your_conf_directory_absolute_path}:/app/conf uscimsc/w4h:latest
  ```