import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import joblib

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('Dataset/Prognostic/data2.csv')
print(data.head())
print("\n")

# Drop unnecessary columns
data.drop(['ID number'], axis=1, inplace=True)
data.replace('?', 0, inplace=True)

# data.replace('?', np.nan, inplace=True)  # Convert '?' to NaN
# data.dropna(axis=0, inplace=True) 

print(data.columns)
print("\n")

print(data.info())
print("\n")

print(data.shape)
print("\n")

print(data.isnull().sum())

print("\n")

print(data.describe())

print("\n")


print(data.describe(include='O').T)
print("\n")

# Encode the target variable
data['Outcome'].replace({"R": 1, "N": 0}, inplace=True)

# plt.figure(figsize=(8, 6))
# sns.countplot(x='Outcome', data=data, palette='viridis')
# plt.title('Prognosis Value Counts')
# plt.xlabel('Prognosis (Recurrent: 1, Non-Recurrent: 0)')
# plt.ylabel('Count')
# plt.show()


fig, ax = plt.subplots(figsize = (5,10))
corr = data.corrwith(data['Outcome']).sort_values(ascending = False).to_frame()
corr.columns = ['Outcome']
sns.heatmap(corr,annot = True,cmap = 'Blues',linewidths = 0.4,linecolor = 'black');
plt.title('Correlation w.r.t Outcome')
plt.tight_layout()
plt.show()

################################################
# Create a copy of the dataframe 
data_copy = data.copy()

# Move 'Outcome' column to the last position
diagnosis_col = data_copy.pop('Outcome')
data_copy['Outcome'] = diagnosis_col

# Save the last 100 rows as a CSV file
d_test = data_copy.head(100)

d_test.drop(columns=['Time','Fractal_dimension_mean','Symmetry_mean','Texture_SE','Concave_points_SE','Concavity_SE','Texture_mean','Symmetry_worst','Smoothness_SE'],inplace=True)
d_test.to_csv('p_test.csv', index=False)
#################################################

##Drop selected columns

data.drop(columns=['Time','Fractal_dimension_mean','Symmetry_mean','Texture_SE','Concave_points_SE','Concavity_SE','Texture_mean','Symmetry_worst','Smoothness_SE'],inplace=True)




# Split the data into features (x) and target variable (y)
x = data.drop(columns='Outcome')
y = data['Outcome']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

print("\n---------------Training Set-------------")
print(x_train.shape)
print(y_train.shape)
print("\n---------------Testing Set--------------")
print(x_test.shape)
print(y_test.shape)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_sample(x_train, y_train)

# Print the shapes of the original and SMOTE-balanced datasets
print("Original Training Data Shape:", x_train.shape, y_train.shape)
print("SMOTE Balanced Training Data Shape:", x_train_smote.shape, y_train_smote.shape)


mms = MinMaxScaler()
x_train_smote=mms.fit_transform(x_train_smote)
x_test=mms.transform(x_test)


# Define individual classifiers
classifier_lr = LogisticRegression(random_state=0, C=10, penalty='l2')
classifier_svc = SVC(kernel='linear', C=0.1)
classifier_tree = DecisionTreeClassifier(random_state=101, criterion='entropy', max_depth=4, min_samples_leaf=3)
classifier_nb = GaussianNB()


# Step 3: Create an ensemble of classifiers
estimators = [
    ('lr', classifier_lr),
    ('svc', classifier_svc),
    ('tree', classifier_tree),
    ('nb', classifier_nb)
]

# Stacking Classifier
ensemble_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Fit the stacking classifier on the training data
ensemble_classifier.fit(x_train_smote, y_train_smote)

# Step 4: Concatenate results from machine learning classifiers
classifier_lr.fit(x_train_smote, y_train_smote)
classifier_svc.fit(x_train_smote, y_train_smote)
classifier_tree.fit(x_train_smote, y_train_smote)
classifier_nb.fit(x_train_smote, y_train_smote)
lr_predictions = classifier_lr.predict(x_test)
svc_predictions = classifier_svc.predict(x_test)
tree_predictions = classifier_tree.predict(x_test)
nb_predictions = classifier_nb.predict(x_test)

# Concatenate predictions
concatenated_predictions = np.column_stack((lr_predictions, svc_predictions, tree_predictions, nb_predictions))
print("Concatenated Predictions:", concatenated_predictions)
print(y)

def model_evaluation(classifier,x_test,y_test):

    cm = confusion_matrix(y_test,classifier.predict(x_test))
    names = ['True Neg','False Pos','False Neg','True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm,annot = labels,cmap = 'Blues',fmt ='')
    plt.show()

    print(classification_report(y_test,classifier.predict(x_test)))

# Step 5: Input the new dataset into the default ANN
# Initialize the ANN model
ann_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=1)

# Fit the ANN model on the concatenated predictions
#ann_model.fit(concatenated_predictions, y_test)
ann_model.fit(x_train_smote, y_train_smote)

# Step 6: Result and evaluation of outputs
# Get predictions from the ensemble model
ensemble_predictions = ensemble_classifier.predict(x_test)

# Evaluate the ensemble model
print("\nEnsemble Model Evaluation:")
model_evaluation(ensemble_classifier, x_test, y_test)

# Calculate ROC_AUC score for the ensemble model
ensemble_roc_auc = roc_auc_score(y_test, ensemble_predictions)
print("ROC_AUC Score for Ensemble Model: {:.2%}".format(ensemble_roc_auc))

# Cross-validation for the ensemble model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
ensemble_cv_score = cross_val_score(ensemble_classifier, x_train_smote, y_train_smote, cv=cv, scoring='roc_auc').mean()
print("\nCross Validation Score for Ensemble Model: {:.2%}".format(ensemble_cv_score))




# # Evaluate the ANN model
# print("\nANN Model Evaluation:")
# model_evaluation(ann_model, concatenated_predictions, y_test)
model_evaluation(ensemble_classifier, x_test, y_test)
model_evaluation(ann_model, x_test, y_test)


# Calculate ROC_AUC score for the ANN model
ann_predictions = ann_model.predict(concatenated_predictions)
ann_roc_auc = roc_auc_score(y_test, ann_predictions)
print("ROC_AUC Score for ANN Model: {:.2%}".format(ann_roc_auc))


joblib.dump(classifier_lr, 'models/Prognostic/classifier_lr_model.joblib')
joblib.dump(classifier_svc, 'models/Prognostic/classifier_svc_model.joblib')
joblib.dump(classifier_tree, 'models/Prognostic/classifier_tree_model.joblib')
joblib.dump(classifier_nb, 'models/Prognostic/classifier_nb_model.joblib')
joblib.dump(ensemble_classifier, 'models/Prognostic/ensemble_classifier_model.joblib')
joblib.dump(ann_model, 'models/Prognostic/ann_model.joblib')
joblib.dump(mms, 'models/Prognostic/minmaxscaler_model.joblib')