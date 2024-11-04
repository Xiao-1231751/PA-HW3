import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

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

# Here is the Feature Ranking by using Information Gain (Entropy Based)

features = df.drop(columns=['Status']) # Get the features columns without Status
target = df['Status'] # Get the Status column

# We need to encoder the vatiable since some variabe are 'While', 'Married'
# Which cannot directly process
for col in features.select_dtypes(include='object').columns:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])

# Call the build in function For Information Gain
information_gain = mutual_info_classif(features, target)

# Get the Feature and Infromation Gain table 
information_gain_features = pd.DataFrame({'Feature': features.columns, 'Information Gain': information_gain})
information_gain_features = information_gain_features.sort_values(by = 'Information Gain', ascending=False)
top_10_features = information_gain_features.head(10)
print(f'The top 0 featutres is {}')


# Here is the Feature Subset Selection 
x = df[top_10_features['Feature'].values]
x = pd.get_dummies(x, drop_first=True)
y = df['Status']

scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

logreg = LogisticRegression()
sfs = SequentialFeatureSelector(logreg, n_features_to_select=5, scoring='accuracy')
sfs.fit(x, y)
selected_features = sfs.get_support()
print('The selected features are:', list(x.columns[selected_features]))



