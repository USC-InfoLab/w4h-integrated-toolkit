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
    cp static/config.yaml.example conf/config.yaml.py
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

to check
## Setup

Run:
   ```shell
   docker run -dp 8501:8501 chickensellerred/w4h:latest 
   ```
Then access http://{your_server_ip}:8501/ to see the dashboard.