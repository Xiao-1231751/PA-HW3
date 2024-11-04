import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor

# # Step 0: Here is the step to read the CSV file and also change the anaplastic; grade iv to 4
# file_path = 'Breast_Cancer_dataset.csv'
# df = pd.read_csv(file_path)

# # Clean the "Grade" column by stripping whitespace and converting to lowercase for comparison
# # If We didn't have this step, it will not work
# df['Grade'] = df['Grade'].str.strip().str.lower()

# # Replace "anaplastic; grade iv" with 4, ignoring any formatting issues
# df['Grade'] = df['Grade'].replace('anaplastic; grade iv', 4)

# df.to_csv('Breast_Cancer_dataset_begin.csv', index=False)
# print("Replacement completed and file saved as Breast_Cancer_dataset_begin.csv.")

# Step 1: Preprocessing 
file_path = 'Breast_Cancer_dataset_step1.csv'
df = pd.read_csv(file_path)
desired_order = [
    'Age', 'Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage', 
    'differentiate', 'Grade', 'A Stage', 'Tumor Size', 'Estrogen Status',
    'Progesterone Status', 'Regional Node Examined', 'Reginol Node Positive', 
    'Survival Months', 'Status'
]

df = df[desired_order]
df.to_csv(file_path, index=False)

# Step 2: Modeling

target_counts = df['Status'].value_counts() # Get the total Status Value
total_rows = len(df) # Get the total rows number
Alive_counts = target_counts['Alive']
Dead_counts = target_counts['Dead']
print(f'Alive counts is {Alive_counts} and Dead counts is {Dead_counts}')

tumor_counts = df['Survival Months']

