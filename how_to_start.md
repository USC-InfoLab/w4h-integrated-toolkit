# Use W4h dashboard to track your patients' health data

## Introduction
The W4H GeoMTS dashboard is designed to provide visualization and analysis capabilities for GeoMTS data catered specifically for Apple Watch demo purposes.

## How to play with it

0. **Log in**  
    We have provided a default account for you to test the system. In the future, w4h will be polished to support multiple users and password management.
    The default account is:
    > username: admin  
      password: admin

1. **Create/Manage your database in your DB instance:**  
At this moment, you should have read the Setting up tutorial and set up your database server.
If you already have data in proper format in your database, you can skip this step.  
Click the "import page" button on the left side of the dashboard, you will see a page like this:  
<img src="./app/static/import_page_create.png" alt="import_page_create" style="width:800px;"/>

    Select "Create new W4H database instance"  
    type in the name of database you want to make.    
    upload your subjects csv file, and check "Populate subject table name", click "Populate database" at the bottom.  
    upload your time series csv file, click "Populate database" at the bottom.  
2. **choose your db in the input page, then play it!**  
    choose your db in the input page  
    select the subjects and control group you want to check
    select if you want to simulate the data
    click "show result"
3. **check the result page**    
 

