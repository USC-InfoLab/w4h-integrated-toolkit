import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from tqdm import tqdm

# Function to generate synthetic time-series data for all features in one pass
def generate_all_features_data(feature_ranges, num_users=100, start_date="2022-01-01", end_date="2022-03-01", freq='5T'):
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    data = { 'user_id': [], 'timestamp': [] }
    for feature in feature_ranges.keys():
        data[feature] = []
    for user_id in tqdm(range(1, num_users + 1)):
        for timestamp in date_range:
            for feature, (value_range, outlier_range) in feature_ranges.items():
                if random.random() < 0.05:  # 5% chance of generating an outlier
                    value = random.uniform(outlier_range[0], outlier_range[1])
                else:
                    value = random.uniform(value_range[0], value_range[1])
                data[feature].append(value)
            data['user_id'].append(user_id)
            data['timestamp'].append(timestamp)
    df = pd.DataFrame(data)
    return df

# Define value ranges and outlier ranges for each feature
feature_ranges = {
    "heart_rate": ((60, 100), (40, 180)),
    "calories": ((0, 100), (0, 300)),
    "mets": ((0, 2), (0, 10)),
    "distances": ((0, 0.1), (0, 1)),
    "steps": ((0, 100), (0, 500)),
    "sleep": ((0, 0.1), (0, 1)),
    "weight": ((100, 200), (80, 250))
}

# Generate synthetic data for all features and write to CSV
df = generate_all_features_data(feature_ranges)
df.to_csv('synthetic_timeseries_data.csv', index=False)
