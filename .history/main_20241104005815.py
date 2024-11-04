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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from C45 import C45Classifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

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

# # KNN 

# # The class structure of KNN
# class KNN:
#     # Initial the KNN class
#     def __init__(self, k):
#         self.k = k
    
#     def calculate_distance(self, point1, point2):
#         return np.sqrt(np.sum((point1 - point2) ** 2))
    
#     def prediction(self, X_test):
#         predictions = []
#         for x in X_test:
#             distances = []
#             # Calculate the distance from x to all other trainPoints
#             for trainPoint in X_train:
#                 dis = self.calculate_distance(x, trainPoint)
#                 distances.append(dis)

#             k_indices = np.argsort(distances)[:self.k] # Get the index of the k closet points 
#             k_nearest_labels = [y_train[i] for i in k_indices] # Get the lanbales with these k cloest points
#             most_common = Counter(k_nearest_labels).most_common(1) # Count the times of label appeared
#             predictions.append(most_common[0][0]) 
#         return predictions
    
# knn_model = KNN(3)
# knn_prediction = knn_model.prediction(X_test)
# # print(f'The knn prediction is {knn_prediction}')

# accuracy = accuracy_score(knn_prediction, y_test)
# recall = recall_score(y_test, knn_prediction, average='weighted')
# precision = precision_score(y_test, knn_prediction, average='weighted')
# recall_none = recall_score(y_test, knn_prediction, average=None)
# precision_none = precision_score(y_test, knn_prediction, average=None)

# print(f"The accuracy for KNN is {accuracy}")
# print(f"The recall for KNN is {recall}")
# print(f"The precision for KNN is {precision}")

# for class_index, (prec, rec) in enumerate(zip(precision_none, recall_none)):
#     print(f"KNN : Class {class_index}: Precision = {prec}, Recall = {rec}")

# print('\n')
# print('\n')


# # Naïve Bayes
# # Default values: priors = None, smoothing = 1e-9
# nb = GaussianNB()
# nb.fit(X_train, y_train)
# nb_prediction = nb.predict(X_test)

# accuracy = accuracy_score(nb_prediction, y_test)
# recall = recall_score(y_test, nb_prediction, average='weighted')
# precision = precision_score(y_test, nb_prediction, average='weighted')
# recall_none = recall_score(y_test, nb_prediction, average=None)
# precision_none = precision_score(y_test, nb_prediction, average=None)

# print(f"The accuracy for Naive Bayes is {accuracy}")
# print(f"The recall for Naive Bayes is {recall}")
# print(f"The precision for Naive Bayes is {precision}")

# for class_index, (prec, rec) in enumerate(zip(precision_none, recall_none)):
#     print(f"Naive Bayes : Class {class_index}: Precision = {prec}, Recall = {rec}")

# print('\n')
# print('\n')
# # https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

# # C4.5 Decision Tree
# # https://towardsdatascience.com/what-is-the-c4-5-algorithm-and-how-does-it-work-2b971a9e7db0
# # https://pypi.org/project/c45-decision-tree/


# # Default values: max_depth = None, min_samples_split = 2, prune_threshold = 0.25, weight = 1
# c45DT = C45Classifier()
# c45DT.fit(X_train, y_train)
# c45DT_prediction = c45DT.predict(X_test)

# accuracy = accuracy_score(c45DT_prediction, y_test)
# recall = recall_score(y_test, c45DT_prediction, average='weighted')
# precision = precision_score(y_test, c45DT_prediction, average='weighted')
# recall_none = recall_score(y_test, c45DT_prediction, average=None)
# precision_none = precision_score(y_test, c45DT_prediction, average=None)

# print(f"The accuracy for C4.5 Decision Tree is {accuracy}")
# print(f"The recall for C4.5 Decision Tree is {recall}")
# print(f"The precision for C4.5 Decision Tree is {precision}")

# for class_index, (prec, rec) in enumerate(zip(precision_none, recall_none)):
#     print(f"C4.5 Decision Tree : Class {class_index}: Precision = {prec}, Recall = {rec}")

# print('\n')
# print('\n')

# # Random Forest

# # Default values: n_estimators = 100, max_depth = None, min_samples_split = 2, 
# # min_sample_leaf = 1, max_features = “sqrt”, bootstrap = True
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# rf_prediction = rf.predict(X_test)

# accuracy = accuracy_score(rf_prediction, y_test)
# recall = recall_score(y_test, rf_prediction, average='weighted')
# precision = precision_score(y_test, rf_prediction, average='weighted')
# recall_none = recall_score(y_test, rf_prediction, average=None)
# precision_none = precision_score(y_test, rf_prediction, average=None)

# print(f"The accuracy for Random Forest is {accuracy}")
# print(f"The recall for Random Forest is {recall}")
# print(f"The precision for Random Forest is {precision}")

# for class_index, (prec, rec) in enumerate(zip(precision_none, recall_none)):
#     print(f"Random Forest : Class {class_index}: Precision = {prec}, Recall = {rec}")

# print('\n')
# print('\n')

# # Gradient Boosting

# # Default values: learning_rate = 0.1, n_estimators = 100, subsample = 1.0, min_samples_split = 2, 
# # min_sample_leaf = 1, max_depth = 3, max_features = None
# gb = GradientBoostingClassifier()
# gb.fit(X_train, y_train)
# gb_prediction = gb.predict(X_test)

# accuracy = accuracy_score(gb_prediction, y_test)
# recall = recall_score(y_test, gb_prediction, average='weighted')
# precision = precision_score(y_test, gb_prediction, average='weighted')
# recall_none = recall_score(y_test, gb_prediction, average=None)
# precision_none = precision_score(y_test, gb_prediction, average=None)

# print(f"The accuracy for Gradient Boosting is {accuracy}")
# print(f"The recall for Gradient Boosting is {recall}")
# print(f"The precision for Gradient Boosting is {precision}")

# for class_index, (prec, rec) in enumerate(zip(precision_none, recall_none)):
#     print(f"Gradient Boosting : Class {class_index}: Precision = {prec}, Recall = {rec}")

# print('\n')
# print('\n')

# # Neural Networks

# # Default values: hidden_layer_sizes = (100,) activation = 'relu', solver = 'adam', alpha = 0.0001, 
# # learning_rate = 'constant', learning_rate_init = 0.001, max_iter = 200, early_stopping = False
# nn = MLPClassifier()
# nn.fit(X_train, y_train)
# nn_prediction = nn.predict(X_test)

# accuracy = accuracy_score(nn_prediction, y_test)
# recall = recall_score(y_test, nn_prediction, average='weighted')
# precision = precision_score(y_test, nn_prediction, average='weighted')
# recall_none = recall_score(y_test, nn_prediction, average=None)
# precision_none = precision_score(y_test, nn_prediction, average=None)

# print(f"The accuracy for Neural Networks is {accuracy}")
# print(f"The recall for Neural Networks is {recall}")
# print(f"The precision for Neural Networks is {precision}")

# for class_index, (prec, rec) in enumerate(zip(precision_none, recall_none)):
#     print(f"Neural Networks : Class {class_index}: Precision = {prec}, Recall = {rec}")






                  ##############     Step 3: Hyperparameter Tuning   ############



# Random Forest

# Default values: n_estimators = 100, max_depth = None, min_samples_split = 2, 
# min_samples_leaf = 1, max_features = “sqrt”, bootstrap = True

# Define model
rf = RandomForestClassifier()
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space_rf = dict()
space_rf['n_estimators'] = [100, 200]
space_rf['max_depth'] = [None, 10, 15]
space_rf['min_samples_split'] = [2, 4, 6]
space_rf['min_samples_leaf'] = [1, 3, 5]
space_rf['max_features'] = ["sqrt", "log2"]
space_rf['bootstrap'] = [True, False]

# define search
search_rf = GridSearchCV(rf, space_rf, scoring='accuracy', n_jobs=-1, cv=cv)
# execute search
result_rf = search_rf.fit(X_train, y_train)

# summarize result
print('Best Score: %s' % result_rf.best_score_)
print('Best Hyperparameters: %s' % result_rf.best_params_)

# Retrieve the best model
best_rf = result_rf.best_estimator_

# Use the best model to make predictions
rf_prediction = best_rf.predict(X_test)

accuracy = accuracy_score(rf_prediction, y_test)
recall = recall_score(y_test, rf_prediction, average='weighted')
precision = precision_score(y_test, rf_prediction, average='weighted')
recall_none = recall_score(y_test, rf_prediction, average=None)
precision_none = precision_score(y_test, rf_prediction, average=None)

print(f"The accuracy for Random Forest is {accuracy}")
print(f"The recall for Random Forest is {recall}")
print(f"The precision for Random Forest is {precision}")

for class_index, (prec, rec) in enumerate(zip(precision_none, recall_none)):
    print(f"Random Forest : Class {class_index}: Precision = {prec}, Recall = {rec}")

print('\n')
print('\n')

# # Gradient Boosting

# Default values: learning_rate = 0.1, n_estimators = 100, subsample = 1.0, min_samples_split = 2, 
# min_sample_leaf = 1, max_depth = 3, max_features = None
gb = GradientBoostingClassifier()

# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space_gb = dict()
space_gb['learning_rate'] = [100, 200]

# define search
search_gb = GridSearchCV(gb, space_gb, scoring='accuracy', n_jobs=-1, cv=cv)
# execute search
result_gb = search_gb.fit(X_train, y_train)

# summarize result
print('Best Score: %s' % result_rf.best_score_)
print('Best Hyperparameters: %s' % result_rf.best_params_)

# Retrieve the best model
best_gb = result_gb.best_estimator_

# Use the best model to make predictions
gb_prediction = best_gb.predict(X_test)

accuracy = accuracy_score(gb_prediction, y_test)
recall = recall_score(y_test, gb_prediction, average='weighted')
precision = precision_score(y_test, gb_prediction, average='weighted')
recall_none = recall_score(y_test, gb_prediction, average=None)
precision_none = precision_score(y_test, gb_prediction, average=None)

print(f"The accuracy for Gradient Boosting is {accuracy}")
print(f"The recall for Gradient Boosting is {recall}")
print(f"The precision for Gradient Boosting is {precision}")

for class_index, (prec, rec) in enumerate(zip(precision_none, recall_none)):
    print(f"Gradient Boosting : Class {class_index}: Precision = {prec}, Recall = {rec}")

print('\n')
print('\n')