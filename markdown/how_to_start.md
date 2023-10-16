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
    <div style="display: flex;justify-content: center; /* 水平居中 */"><img src="./app/static/import_page_create.png" alt="import_page_create" style="margin:10px;width:800px;border: 2px solid rgba(252, 176, 69, 0.3);"/></div>  

    Let's start from creating a new database instance:  
    First, Select "Create new W4H database instance"    
    <div style="display: flex;justify-content: center; /* 水平居中 */"><img src="./app/static/create_new_db.png" alt="create_new_db" style="margin:10px;width:800px;border: 2px solid rgba(252, 176, 69, 0.3);"/></div>  
    
    Second, type in the name of database you want to make, and click "create". In this case we name it "test2"
    <div style="display: flex;justify-content: center; /* 水平居中 */"><img src="./app/static/set_db_name.png" alt="set_db_name" style="margin:10px;width:800px;border: 2px solid rgba(252, 176, 69, 0.3);"/></div>  
    If it's created successfully, you will see a message like this:  
    <div style="display: flex;justify-content: center; /* 水平居中 */"><img src="./app/static/create_success.png" alt="create_success" style="margin:10px;width:800px;border: 2px solid rgba(252, 176, 69, 0.3);"/></div>  
    
    Third, select "Choose existing W4H database instance"  
    <div style="display: flex;justify-content: center; /* 水平居中 */"><img src="./app/static/choose_exist_db.png" alt="choose_exist_db" style="margin:10px;width:800px;border: 2px solid rgba(252, 176, 69, 0.3);"/></div>  
    Select the database you just created  
    <div style="display: flex;justify-content: center; /* 水平居中 */"><img src="./app/static/select_exist_db.png" alt="select_exist_db" style="margin:10px;width:800px;border: 2px solid rgba(252, 176, 69, 0.3);"/></div>

    For the Following step, if you need some test file, try to download here:  
    [synthetic_subject_data.csv](../app/static/synthetic_subject_data.csv)  
    [synthetic_timeseries_data_reduced.csv](../app/static/synthetic_timeseries_data_reduced.csv)  

    Fourth, Upload your subjects csv file, and check "Populate subject table name".
    <div style="display: flex;justify-content: center; /* 水平居中 */"><img src="./app/static/upload_subject_csv.png" alt="upload_subject_csv" style="margin:10px;width:800px;border: 2px solid rgba(252, 176, 69, 0.3);"/></div>  
    After making sure corresponding Column are all correct, click "Populate database" at the bottom.  
    <div style="display: flex;justify-content: center; /* 水平居中 */"><img src="./app/static/populate_db.png" alt="populate_db" style="margin:10px;width:800px;border: 2px solid rgba(252, 176, 69, 0.3);"/></div>

    Fifth, upload your time series csv file  
    <div style="display: flex;justify-content: center; /* 水平居中 */"><img src="./app/static/upload_time_csv.png" alt="upload_time_csv" style="margin:10px;width:800px;border: 2px solid rgba(252, 176, 69, 0.3);"/></div>  
    After making sure corresponding Column are all correct, click "Populate database" at the bottom.  
    <div style="display: flex;justify-content: center; /* 水平居中 */"><img src="./app/static/populate_db_time.png" alt="populate_db_time" style="margin:10px;width:800px;border: 2px solid rgba(252, 176, 69, 0.3);"/></div>  
    
    (Optional)Sixth, open your DB management tool, such as PgAdmin4, and check if the data is populated correctly.  
    <div style="display: flex;justify-content: center; /* 水平居中 */"><img src="./app/static/pgadmin.png" alt="pgadmin" style="margin:10px;width:800px;border: 2px solid rgba(252, 176, 69, 0.3);"/></div>  

2. **choose your db in the input page, then setup it!**  
    
    choose your db in the input page  
    <div style="display: flex;justify-content: center; /* 水平居中 */"><img src="./app/static/input_select_db.png" alt="input_select_db" style="margin:10px;width:800px;border: 2px solid rgba(252, 176, 69, 0.3);"/></div>   

    select the subjects and control group you want to check  
    <div style="display: flex;justify-content: center; /* 水平居中 */"><img src="./app/static/subjects_and_control_group.png" alt="subjects_and_control_group" style="margin:10px;width:800px;border: 2px solid rgba(252, 176, 69, 0.3);"/></div>  

    select if you want to simulate the data
    click "show result"
3. **check the result page**  
    You are there!  
    <div style="display: flex;justify-content: center; /* 水平居中 */"><img src="./app/static/result_page.png" alt="result_page" style="margin:10px;width:800px;border: 2px solid rgba(252, 176, 69, 0.3);"/></div>  


