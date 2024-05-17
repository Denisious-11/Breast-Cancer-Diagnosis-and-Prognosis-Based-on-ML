import csv

# Reading data from the file
file_path = 'wpbc.data'
with open(file_path, 'r') as file:
    data = file.read()

# Splitting the data into lines and extracting headers
lines = data.strip().split('\n')
headers = ["ID number", "Outcome", "Time", "Radius_mean", "Texture_mean", "Perimeter_mean", "Area_mean", "Smoothness_mean", "Compactness_mean", "Concavity_mean", "Concave_points_mean", "Symmetry_mean", "Fractal_dimension_mean", "Radius_SE", "Texture_SE", "Perimeter_SE", "Area_SE", "Smoothness_SE", "Compactness_SE", "Concavity_SE", "Concave_points_SE", "Symmetry_SE", "Fractal_dimension_SE", "Radius_worst", "Texture_worst", "Perimeter_worst", "Area_worst", "Smoothness_worst", "Compactness_worst", "Concavity_worst", "Concave_points_worst", "Symmetry_worst", "Fractal_dimension_worst", "Tumor_size", "Lymph_node_status"]

# Creating CSV file
csv_filename = 'data2.csv'
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Writing headers
    writer.writerow(headers)
    
    # Writing data
    for line in lines:
        row = line.split(',')
        writer.writerow(row)
    
print(f"CSV file '{csv_filename}' created successfully.")
