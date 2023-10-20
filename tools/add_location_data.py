import pandas as pd
import random
import math

import folium

count = 0
def generate_point(start_y = 34.0224, start_x = -118.2851):
    # Define a radius in meters around the USC location
    usc_radius = 5000

    # Define the Earth's radius in meters
    R = 6371000

    # Maximum distance at each interval
    max_dist = 10000
    global count
    count += 1
    r = usc_radius / math.sqrt(random.uniform(0, 1))
    theta = random.uniform(0, 2*math.pi)
    new_start_x = start_x + (r / R) * math.cos(theta)
    new_start_y = start_y + (r / R) * math.sin(theta)
    # print(count,'(', start_y, start_x, ') (',new_start_y, new_start_x,')')
    return (new_start_y, new_start_x)



df = pd.read_csv('combined_msband.csv')
df['location'] = None
pre_y = -1.0
pre_x = -1.0
for index, row in df.iterrows():
    email = row['email']

    if index == 0 or email != df.at[index - 1, 'email']:
        current_location = generate_point()
    else:
        current_location = generate_point(pre_y, pre_x)

    # 将坐标更新到 "location" 列
    df.at[index, 'location'] = current_location
    pre_y = current_location[0]
    pre_x = current_location[1]

# Create a map centered at the USC location
m = folium.Map(location=(34.0224,-118.2851), zoom_start=15)

folium.Marker(location=df.at[0,'location'], icon=folium.Icon(color='green')).add_to(m)
folium.Marker(location=df.at[100,'location'], icon=folium.Icon(color='red')).add_to(m)

# Add a line representing the person's trajectory
folium.PolyLine(locations=df['location'].iloc[:100], color='blue').add_to(m)
# m.save('map.html')
# print(df)

df.to_csv('combined_msband_location.csv', index=False)