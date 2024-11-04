import pandas as pd

file_path = 'Breast_Cancer_dataset.csv'
df = pd.read_csv(file_path)

# Replace "anaplastic; Grade IV" with 4 in the relevant column
# Replace 'YourColumnName' with the actual column name that contains this value
df['Grade'] = df['Grade'].replace('anaplastic; Grade IV', 4)

df.to_csv('Breast_Cancer_dataset_begin.csv', index=False)

print("Replacement completed and file saved as Breast_Cancer_dataset_begin.csv.")
