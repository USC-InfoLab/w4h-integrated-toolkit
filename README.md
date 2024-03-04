# W4H Integrated Toolkit
The Wearables for Health (W4H) Integrated Toolkit Repository serves as a centralized hub for managing, analyzing, and visualizing wearable data in a seamless and efficient manner. This repository hosts a suite of open-source tools and frameworks meticulously engineered to facilitate end-to-end processing of wearable data, ranging from data ingestion from various sources to real-time and offline analytics, all the way to insightful visualization. At the core of the toolkit lies the novel Geospatial Multivariate Time Series (GeoMTS) abstraction, which enables streamlined management and analysis of wearable data. The repository encompasses key components such as StreamSim for simulating real-time data streaming, W4H ImportHub for integrating offline datasets, pyGarminAPI and pyFitbitAPI for efficient interaction with popular wearable device APIs, and an Integrated Analytics Dashboard for effective data extraction, presentation, and analysis.

## Toolkit Components
This toolkit comprises the following tools, each tailored to address specific needs in the wearable data lifecycle:

- **[StreamSim](https://github.com/USC-InfoLab/StreamSim)**: A real-time data streaming simulator tool for tabular data, aiding in testing real-time applications.

- **[W4H ImportHub](https://github.com/USC-InfoLab/W4H-ImportHub)**: Your gateway for seamlessly integrating stored datasets like CSV files into the W4H platform.

- **[pyGarminAPI](https://github.com/USC-InfoLab/pyGarminAPI)**: A Python library to interact with the Garmin API, facilitating the retrieval of activity and health data from Garmin devices.

- **Integrated Analytics Dashboard**: Core component for GeoMTS data extraction, presentation, and analysis, supporting both real-time and offline analytics on GeoMTS data. (Source code included in this repo)

- **[Approximate Aggregate Queries on GeoMTS](https://github.com/USC-InfoLab/fft_approximate)**: A PostgreSQL extension `fft_approximate` for quickly answering aggregate range queries on time series data using an approximate approach.


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
   docker run -dp 8501:8501 uscimsc/w4h:latest 
   ```
Then access http://{your_server_ip}:8501/ to see the dashboard.
