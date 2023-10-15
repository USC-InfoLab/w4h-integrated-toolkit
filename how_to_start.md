# Use W4h dashboard to track your patients' health data

## Introduction
The W4H GeoMTS dashboard is designed to provide visualization and analysis capabilities for GeoMTS data catered specifically for Apple Watch demo purposes.

## How to play with it

0. **Log in**  
    We have provided a default account for you to test the system. In the future, w4h will be polished to support multiple users and password management.
    The default account is:
    > username: admin  
      password: admin

1. **Create/Manage your database instance in your DB server:**  
At this moment, you should have read the Setting up tutorial and set up your database server.
If you already have data in proper format in your database, you can skip this step.  
Click the "import page" button on the left side of the dashboard, you will see a page like this:  
<img src="./app/static/import_page_create.png" alt="import_page_create" style="width:800px;"/>  

    Let's start from creating a new database instance:  
    First, Select "Create new W4H database instance"  
    <div style="display: flex;justify-content: center; /* 水平居中 */"><img src="./app/static/create_new_db.png" alt="create_new_db" style="margin:10px;width:800px;border: 2px solid rgba(252, 176, 69, 0.5);"/></div>  
    
    Second, type in the name of database you want to make, and click "create". In this case we name it "test2"
    <img src="./app/static/set_db_name.png" alt="set_db_name" style="width:800px;"/>  
    If it's created successfully, you will see a message like this:  
    <img src="./app/static/create_success.png" alt="create_success" style="width:800px;"/>  
    
    Third, select "Choose existing W4H database instance"  
    <img src="./app/static/choose_exist_db.png" alt="choose_exist_db" style="width:800px;"/>  
    Select the database you just created  
    <img src="./app/static/select_exist_db.png" alt="select_exist_db" style="width:800px;"/>    
    
    Fourth, Upload your subjects csv file, and check "Populate subject table name".  
    <img src="./app/static/upload_subject_csv.png" alt="upload_subject_csv" style="width:800px;"/>  
    After making sure corresponding Column are all correct, click "Populate database" at the bottom.  
    <img src="./app/static/populate_db.png" alt="populate_db" style="width:800px;"/>

    Fifth, upload your time series csv file  
    <img src="./app/static/upload_time_csv.png" alt="upload_time_csv" style="width:800px;"/>  
    After making sure corresponding Column are all correct, click "Populate database" at the bottom.  
    <img src="./app/static/populate_db_time.png" alt="populate_db_time" style="width:800px;"/>  
    
    (Optional)Sixth, open your DB management tool, such as PgAdmin4, and check if the data is populated correctly.  
    <img src="./app/static/pgadmin.png" alt="pgadmin" style="width:800px;"/>  

2. **choose your db in the input page, then setup it!**  
    
    choose your db in the input page  
    <img src="./app/static/input_select_db.png" alt="input_select_db" style="width:800px;"/>   

    select the subjects and control group you want to check  
    <img src="./app/static/subjects_and_control_group.png" alt="subjects_and_control_group" style="width:800px;"/>  

    select if you want to simulate the data
    click "show result"
3. **check the result page**  
    You are there!  
    <img src="./app/static/result_page.png" alt="result_page" style="width:800px;"/>  


