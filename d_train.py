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
import joblib

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('Dataset/Diagnostic/data.csv')
print(data.head())
print("\n")

# Drop unnecessary columns
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

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
data['diagnosis'].replace({"M": 1, "B": 0}, inplace=True)

# plt.figure(figsize=(8, 6))
# sns.countplot(x='diagnosis', data=data, palette='viridis')
# plt.title('Diagnosis Value Counts')
# plt.xlabel('Diagnosis (Malignant: 1, Benign: 0)')
# plt.ylabel('Count')
# plt.show()

################################################
# Create a copy of the dataframe 
data_copy = data.copy()

# Move 'diagnosis' column to the last position
diagnosis_col = data_copy.pop('diagnosis')
data_copy['diagnosis'] = diagnosis_col

# Save the last 100 rows as a CSV file
d_test = data_copy.head(100)
d_test.drop(columns=['fractal_dimension_se', 'smoothness_se', 'symmetry_se', 'texture_se', 'fractal_dimension_mean'],
          inplace=True)
### d_test.to_csv('d_test.csv', index=False)
#################################################

# Drop selected columns
data.drop(columns=['fractal_dimension_se', 'smoothness_se', 'symmetry_se', 'texture_se', 'fractal_dimension_mean'],
          inplace=True)



# fig, ax = plt.subplots(figsize = (5,10))
# corr = data.corrwith(data['diagnosis']).sort_values(ascending = False).to_frame()
# corr.columns = ['diagnosis']
# sns.heatmap(corr,annot = True,cmap = 'Blues',linewidths = 0.4,linecolor = 'black');
# plt.title('Correlation w.r.t diagnosis')
# plt.tight_layout()
# plt.show()


# Split the data into features (x) and target variable (y)
x = data.drop(columns='diagnosis')
y = data['diagnosis']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

print("\n---------------Training Set-------------")
print(x_train.shape)
print(y_train.shape)
print("\n---------------Testing Set--------------")
print(x_test.shape)
print(y_test.shape)

# Normalize the features
mms = MinMaxScaler()
x_train=mms.fit_transform(x_train)
x_test=mms.transform(x_test)

print("\n")

print(data.head())

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
ensemble_classifier.fit(x_train, y_train)

# Step 4: Concatenate results from machine learning classifiers
classifier_lr.fit(x_train, y_train)
classifier_svc.fit(x_train, y_train)
classifier_tree.fit(x_train, y_train)
classifier_nb.fit(x_train, y_train)
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
ann_model.fit(concatenated_predictions, y_test)
########ann_model.fit(x_train, y_train)

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
ensemble_cv_score = cross_val_score(ensemble_classifier, x_train, y_train, cv=cv, scoring='roc_auc').mean()
print("\nCross Validation Score for Ensemble Model: {:.2%}".format(ensemble_cv_score))




# Evaluate the ANN model
print("\nANN Model Evaluation:")
model_evaluation(ann_model, concatenated_predictions, y_test)

# Calculate ROC_AUC score for the ANN model
ann_predictions = ann_model.predict(concatenated_predictions)
ann_roc_auc = roc_auc_score(y_test, ann_predictions)
print("ROC_AUC Score for ANN Model: {:.2%}".format(ann_roc_auc))


joblib.dump(classifier_lr, 'models/Diagnostic/classifier_lr_model.joblib')
joblib.dump(classifier_svc, 'models/Diagnostic/classifier_svc_model.joblib')
joblib.dump(classifier_tree, 'models/Diagnostic/classifier_tree_model.joblib')
joblib.dump(classifier_nb, 'models/Diagnostic/classifier_nb_model.joblib')
joblib.dump(ensemble_classifier, 'models/Diagnostic/ensemble_classifier_model.joblib')
joblib.dump(ann_model, 'models/Diagnostic/ann_model.joblib')
joblib.dump(mms, 'models/Diagnostic/minmaxscaler_model.joblib')