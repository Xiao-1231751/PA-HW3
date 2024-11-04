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


# Step 1: 
# Load the dataset
file_path = 'Breast_Cancer_dataset_begin.csv'  # Update to your file path
df = pd.read_csv(file_path)

# Separate numeric and non-numeric columns
numeric_df = df.select_dtypes(include=[np.number])
non_numeric_df = df.select_dtypes(exclude=[np.number])

# Step 1: Handle Missing Values by Replacing with Column Mean (only for numeric columns)
imputer = SimpleImputer(strategy='mean')
numeric_df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)

# Concatenate numeric and non-numeric data back together
df_imputed = pd.concat([numeric_df_imputed, non_numeric_df.reset_index(drop=True)], axis=1)

# Step 2: Use Local Outlier Factor to Identify Outliers (only on numeric columns)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)  # Adjust contamination to expected % of outliers
lof_labels = lof.fit_predict(numeric_df_imputed)

# Identify outliers based on LOF score
# Outliers are labeled as -1 by LOF, inliers as 1
outliers = numeric_df_imputed[lof_labels == -1]
inliers = numeric_df_imputed[lof_labels == 1]

# Optional: Print or inspect outliers
print("Number of Outliers Detected:", outliers.shape[0])
print("Outliers:\n", outliers)

# Save the imputed dataset to a new file (including outliers)
df_imputed.to_csv('Breast_Cancer_dataset_imputed.csv', index=False)

print("Imputed data saved to 'Breast_Cancer_dataset_imputed.csv'.")
