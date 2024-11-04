import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score, precision_score
from sklearn.naive_bayes import GaussianNB

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
# print(f'The top 10 featutres is {top_10_features}')

# Here is the Feature Subset Selection 
x = df[top_10_features['Feature'].values]
y = df['Status']

for col in x.select_dtypes(include='object').columns:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])

scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

logreg = LogisticRegression()
sfs = SequentialFeatureSelector(logreg, n_features_to_select=5, scoring='accuracy')
sfs.fit(x, y)
selected_features = list(x.columns[sfs.get_support()])
# print('The selected features are:', selected_features)

# Get the train data and test data
X = df[selected_features]

for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y.astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104, test_size=0.30, shuffle=True)
print('X_train : ') 
print(X_train.head()) 
print('') 
print('X_test : ') 
print(X_test.head()) 
print('') 
print('y_train : ') 
print(y_train) 
print('') 
print('y_test : ') 
print(y_test)

X_train = np.array(X_train)
X_test = np.array(X_test)

# KNN 

# The class structure of KNN
class KNN:
    # Initial the KNN class
    def __init__(self, k):
        self.k = k
    
    def calculate_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def prediction(self, X_test):
        predictions = []
        for x in X_test:
            distances = []
            # Calculate the distance from x to all other trainPoints
            for trainPoint in X_train:
                dis = self.calculate_distance(x, trainPoint)
                distances.append(dis)

            k_indices = np.argsort(distances)[:self.k] # Get the index of the k closet points 
            k_nearest_labels = [y_train[i] for i in k_indices] # Get the lanbales with these k cloest points
            most_common = Counter(k_nearest_labels).most_common(1) # Count the times of label appeared
            predictions.append(most_common[0][0]) 
        return predictions
    
knn_model = KNN(3)
knn_prediction = knn_model.prediction(X_test)
# print(f'The knn prediction is {knn_prediction}')

accuracy = accuracy_score(knn_prediction, y_test)
recall = recall_score(y_test, knn_prediction, average='weighted')
precision = precision_score(y_test, knn_prediction, average='weighted')
recall_none = recall_score(y_test, knn_prediction, average=None)
precision_none = precision_score(y_test, knn_prediction, average=None)

print(f"The accuracy for KNN is {accuracy}")
print(f"The recall for KNN is {recall}")
print(f"The precision for KNN is {precision}")

for class_index, (prec, rec) in enumerate(zip(precision_none, recall_none)):
    print(f"Class {class_index}: Precision = {prec}, Recall = {rec}")


# Naïve Bayes

nb = GaussianNB()
nb.fit(X_train, y_train)
nb_prediction = nb.predict

