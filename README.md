# W4H GeoMTS Dashboard for Apple Watch Demo
The W4H GeoMTS dashboard is designed to provide visualization and analysis capabilities for GeoMTS data catered specifically for Apple Watch demo purposes.

## Prerequisites
Ensure you have `python` and `pip` installed on your machine.

## Setup
1. **Install Required Packages:**
First, navigate to the project directory and install the necessary packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

2. **Configure Database Properties:**
Copy the provided example configuration file to create your own configuration:
    ```bash
    cp conf.py.example conf.py
    ```
    Now, edit the `conf.py` file with your desired database properties and credentials. Ensure you have proper access rights and credentials for the database.

3. **Stream Simulation:**
To start the stream simulation service, run the following command:
    ```bash
    python stream_sim.py
    ```

4. **Start the Dashboard:**
After the stream simulation service is up and running, initiate the dashboard using `streamlit`:
    ```bash
    streamlit run viz.py
    ```
    Once the dashboard is started, you can access it via the URL provided by `streamlit` in your terminal.

# Setup your server with Docker  

## Prerequisites
docker has been installed in you server. run:  
```shell
docker
```

to see
## Setup

1. **test version of w4h:**
It includes a default test database with sample data. you can setup it and see how the system works without your own data source.  
Run:
   ```shell
   docker run -dp 8501:8501 chickensellerred/w4h:test 
   ```

   And you will see a login portal, you can use the default account to test it:  
   >username: admin  
   password: admin
   
2. **Use your own database:**
If you have your own database(postgreSQL), you can create a conf directory, and put your conf.py in it.  
like this:
   >.  
   |____conf  
   | |____conf.py

   You can refer to conf.py.example to write conf.py
   Now you're ready to set the docker!  
   Run:
   ```shell
   docker run -dp 8501:8501 -v {your_conf_directory_absolute_path}:app/conf/ chickensellerred/w4h:1.0
   ```