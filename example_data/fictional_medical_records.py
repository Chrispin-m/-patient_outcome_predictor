import pandas as pd
from faker import Faker
import random

fake = Faker()

def generate_medical_condition():
    return ', '.join(fake.words(3))

data = {
    'Patient ID': [fake.random_int(min=1, max=1000) for _ in range(100)],
    'Name': [fake.name() for _ in range(100)],
    'Date of birth': [fake.date_of_birth(minimum_age=1, maximum_age=100) for _ in range(100)],
    'Gender': [random.choice(['M', 'F']) for _ in range(100)],
    'Medical conditions': [generate_medical_condition() for _ in range(100)],
    'Medications': [generate_medical_condition() for _ in range(100)],
    'Allergies': [generate_medical_condition() for _ in range(100)],
    'Last appointment date': [fake.date_between(start_date='-2y', end_date='today') for _ in range(100)],
    'outcome': [random.choice([0, 1]) for _ in range(100)]
}

df = pd.DataFrame(data)
df.to_csv('example_data/fictional_medical_records.csv', index=False)
