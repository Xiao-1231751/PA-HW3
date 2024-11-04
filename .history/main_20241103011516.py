import pandas as pd

# Load the dataset
file_path = 'Breast_Cancer_dataset.csv'
df = pd.read_csv(file_path)

# Clean the "Grade" column by stripping whitespace and converting to lowercase for comparison
df['Grade'] = df['Grade'].str.strip().str.lower()

# Replace "anaplastic; grade iv" with 4, ignoring any formatting issues
df['Grade'] = df['Grade'].replace('anaplastic; grade iv', 4)

# Save the updated dataset to a new file
df.to_csv('Breast_Cancer_dataset_begin.csv', index=False)

print("Replacement completed and file saved as Breast_Cancer_dataset_begin.csv.")
