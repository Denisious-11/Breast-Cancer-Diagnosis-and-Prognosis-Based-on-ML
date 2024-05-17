import numpy as np
import pandas as pd
import joblib

# Provide the values as a list
new_data_values = [27.22,182.1,2250.0,0.1094,0.1914,0.2871,0.1878,0.8361,5.82,128.7,0.02537,0.01575,0.002747,33.12,32.85,220.8,3216.0,0.1472,0.4034,0.534,0.2688,0.08082,4.0,4]

# Reshape the list into a 2D array (needed for scaling)
new_data_array = np.array(new_data_values).reshape(1, -1)

# Create a DataFrame using the reshaped array
new_data = pd.DataFrame(new_data_array, columns=[
    'Radius_mean','Perimeter_mean','Area_mean','Smoothness_mean','Compactness_mean','Concavity_mean',
    'Concave_points_mean','Radius_SE','Perimeter_SE','Area_SE','Compactness_SE','Symmetry_SE',
    'Fractal_dimension_SE','Radius_worst','Texture_worst','Perimeter_worst','Area_worst',
    'Smoothness_worst','Compactness_worst','Concavity_worst','Concave_points_worst','Fractal_dimension_worst',
    'Tumor_size','Lymph_node_status'
])

# Load the MinMaxScaler
mms = joblib.load('models/Prognostic/minmaxscaler_model_.joblib')

# Scale the new data using the MinMaxScaler
cp = mms.transform(new_data)


# Load the classifiers
classifier_lr = joblib.load('models/Prognostic/classifier_lr_model.joblib')
classifier_svc = joblib.load('models/Prognostic/classifier_svc_model.joblib')
classifier_tree = joblib.load('models/Prognostic/classifier_tree_model.joblib')
classifier_nb = joblib.load('models/Prognostic/classifier_nb_model.joblib')
ensemble_classifier = joblib.load('models/Prognostic/ensemble_classifier_model.joblib')

# Make predictions using classifiers
lr_predictions = classifier_lr.predict(cp)
svc_predictions = classifier_svc.predict(cp)
tree_predictions = classifier_tree.predict(cp)
nb_predictions = classifier_nb.predict(cp)

# Concatenate predictions
cp_ = np.column_stack((lr_predictions, svc_predictions, tree_predictions, nb_predictions))
print("Concatenated Predictions:", cp_)

# Load the ANN model
ann_model = joblib.load('models/Prognostic/ann_model_.joblib')

# Make predictions using the ANN model
ann_predictions = ann_model.predict(cp)

# Print or use the predictions as needed

print("ANN Predictions:", ann_predictions)
