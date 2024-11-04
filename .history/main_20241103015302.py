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
file_path = 'Breast_Cancer_dataset.csv'  # Update to your file path
df = pd.read_csv(file_path)

# Step 1: Handle Missing Values by Replacing with Column Mean
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Step 2: Use Local Outlier Factor to Identify Outliers
# Selecting only numeric columns for LOF, as LOF works with numerical data
numeric_df = df_imputed.select_dtypes(include=[np.number])

# Initialize LOF with appropriate parameters (default neighbors=20, change if necessary)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)  # Adjust contamination to expected % of outliers
lof_labels = lof.fit_predict(numeric_df)

# Identify outliers based on LOF score
# Outliers are labeled as -1 by LOF, inliers as 1
outliers = numeric_df[lof_labels == -1]
inliers = numeric_df[lof_labels == 1]

# Optional: Print or inspect outliers
print("Number of Outliers Detected:", outliers.shape[0])
print("Outliers:\n", outliers)

# Save the results (inliers only, excluding outliers if desired)
# Uncomment the following line if you want to save inliers without outliers
# inliers.to_csv('Breast_Cancer_dataset_no_outliers.csv', index=False)

# Save the modified dataset with imputed values to a new file (including outliers)
df_imputed.to_csv('Breast_Cancer_dataset_imputed.csv', index=False)

print("Imputed data saved to 'Breast_Cancer_dataset_imputed.csv'")
print("Inliers (without outliers) saved to 'Breast_Cancer_dataset_no_outliers.csv' (if uncommented).")





