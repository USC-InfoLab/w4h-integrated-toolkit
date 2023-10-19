import pandas as pd
from faker import Faker
import random
import uuid

# Initialize Faker generator
fake = Faker()

def generate_synthetic_subject_data(num_records, unique_patients_list):
    data = []
    for i in range(1, num_records + 1):
        record = [
            i,  # id
            unique_patients_list[i-1], #str(uuid.uuid4()),  # subj_id
            random.randint(1, 20),  # device_id
            random.randint(1, 2),  # cohort_id
            fake.boolean(),  # dropped
            "Pass" if fake.boolean() else "DOR",  # drop_type
            fake.random_element(elements=("SGT", "HN", "PFC")),  # rank
            fake.date_between(start_date="-10y", end_date="today").strftime('%Y-%m-%d'),  # rank_date
            fake.random_element(elements=("ALAT", "NAV", "MCT/HZ", "UH")),  # pref
            fake.random_int(min=20180000, max=20190000) if fake.boolean() else "NA",  # rtc_checkin_date
            random.randint(1, 3),  # brpc_attempts
            fake.state_abbr(),  # pop
            fake.random_int(min=100, max=999),  # mos
            fake.date_between(start_date="-10y", end_date="today").strftime('%Y-%m-%d'),  # pebd
            (fake.date_between(start_date="-10y", end_date="today") + pd.DateOffset(years=10)).strftime('%Y-%m-%d'),  # eas
            fake.date_of_birth(minimum_age=20, maximum_age=50).strftime('%Y-%m-%d'),  # dob
            random.randint(60, 75),  # ht
            random.randint(130, 200),  # wt
            fake.random_int(min=100, max=999),  # area_code
            "N",  # married
            fake.city(),  # city
            fake.state_abbr(),  # state
            fake.company(),  # hs_name
            fake.city(),  # hs_city
            fake.state_abbr(),  # hs_state
            random.randint(2000, 2020),  # hs_grad_year
            random.randint(0, 4),  # hs_sports
            random.randint(0, 4),  # hs_clubs
            fake.company() if fake.boolean() else "NA",  # college_name
            fake.city() if fake.boolean() else "NA",  # college_city
            fake.state_abbr() if fake.boolean() else "NA",  # college_state
            random.randint(2000, 2020) if fake.boolean() else "NA",  # college_grad_year
            random.randint(0, 4) if fake.boolean() else "NA",  # college_sports
            random.randint(0, 4) if fake.boolean() else "NA",  # college_clubs
            fake.sentence(),  # hobbies
            random.randint(1, 3),  # swimming_exp
            random.randint(1, 3),  # workout_exp
            fake.time() if fake.boolean() else "NA",  # DOR_Time
            fake.random_letter().upper() if fake.boolean() else "NA",  # DOR_Activity
            fake.date_between(start_date="-10y", end_date="today").strftime('%Y-%m-%d') if fake.boolean() else "NA",  # drop_date
            random.randint(20, 50),  # age
            random.randint(55, 75),  # height
            random.randint(130, 200)  # weight
        ]
        data.append(record)
    return data

# Generate synthetic data
Origindf = pd.read_csv('combined_msband.csv')
# 对"Patient"列进行唯一操作并将唯一值放入一个数组
unique_patients = Origindf['email'].unique()

# 如果你想将唯一值存储在一个列表中，可以使用tolist()方法
unique_patients_list = unique_patients.tolist()
print(unique_patients_list)



num_records = len(unique_patients_list)  # Specify the number of records you want to generate
synthetic_data = generate_synthetic_subject_data(num_records,unique_patients_list)

# Create DataFrame
columns = [
    "user_id", "subj_id", "device_id", "cohort_id", "dropped", "drop_type", "rank", "rank_date", "pref", "rtc_checkin_date",
    "brpc_attempts", "pop", "mos", "pebd", "eas", "dob", "ht", "wt", "area_code", "married", "city", "state", "hs_name",
    "hs_city", "hs_state", "hs_grad_year", "hs_sports", "hs_clubs", "college_name", "college_city", "college_state",
    "college_grad_year", "college_sports", "college_clubs", "hobbies", "swimming_exp", "workout_exp", "DOR_Time",
    "DOR_Activity", "drop_date", "age", "height", "weight"
]
df = pd.DataFrame(synthetic_data, columns=columns)

# Write to CSV
df.to_csv('combined_subject_msband.csv', index=False)