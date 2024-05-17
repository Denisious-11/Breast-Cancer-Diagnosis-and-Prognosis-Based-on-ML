import numpy as np
import pandas as pd
import joblib

# Provide the values as a list
new_data_values = [8.597999999999999,20.98,54.66,221.8,0.1243,0.08963,0.03,0.009259,0.1828,0.3582,2.4930000000000003,18.39,0.03162,0.03,0.009259,9.565,27.04,62.06,273.9,0.1639,0.1698,0.09001,0.027780000000000003,0.2972,0.07712000000000001]

# Reshape the list into a 2D array (needed for scaling)
new_data_array = np.array(new_data_values).reshape(1, -1)

# Create a DataFrame using the reshaped array
new_data = pd.DataFrame(new_data_array, columns=[
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
    'concavity_mean', 'concave points_mean', 'symmetry_mean', 'radius_se', 
    'perimeter_se', 'area_se', 'compactness_se', 'concavity_se', 'concave points_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
])

# Load the MinMaxScaler
mms = joblib.load('models/Diagnostic/minmaxscaler_model.joblib')

# Scale the new data using the MinMaxScaler
cp = mms.transform(new_data)


# Load the classifiers
classifier_lr = joblib.load('models/Diagnostic/classifier_lr_model.joblib')
classifier_svc = joblib.load('models/Diagnostic/classifier_svc_model.joblib')
classifier_tree = joblib.load('models/Diagnostic/classifier_tree_model.joblib')
classifier_nb = joblib.load('models/Diagnostic/classifier_nb_model.joblib')
ensemble_classifier = joblib.load('models/Diagnostic/ensemble_classifier_model.joblib')

# Make predictions using classifiers
lr_predictions = classifier_lr.predict(cp)
svc_predictions = classifier_svc.predict(cp)
tree_predictions = classifier_tree.predict(cp)
nb_predictions = classifier_nb.predict(cp)

# Concatenate predictions
cp_ = np.column_stack((lr_predictions, svc_predictions, tree_predictions, nb_predictions))
print("Concatenated Predictions:", cp_)

# Load the ANN model
ann_model = joblib.load('models/Diagnostic/ann_model.joblib')

# Make predictions using the ANN model
ann_predictions = ann_model.predict(cp_)

# Print or use the predictions as needed

print("ANN Predictions:", ann_predictions)
