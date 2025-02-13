"""
This script computes the subsystems LKIF calling the function from file
"compute_liang_subsystems". Also implements p-value analysis
"""

import numpy as np
import csv
import time
import os 

# add things for path if you move files around 
from compute_liang_subsystems import information_flow_subspace

# Start timer
start = time.time()

# Load time series data
accelerate = 1
file_path = '/home/chiaraz/data_thesis/data_1e5points_1000ws/window_for_TE/avg_atmo/'
file_name = '100yr_weak_largewindow'
file = file_path + file_name
#file_path = '/home/chiaraz/data_thesis/data_1e5points_1000ws/evol_fields_1e-8.dat'
data_in_file = []

with open(file, 'r') as file:
    for index, line in enumerate(file):
        if index % accelerate == 0:
            try:
                row = [float(value) for value in line.split()]
                if len(row) < 37:
                    print(f"Line has insufficient data: {line.strip()}")
                    continue
                data_in_file.append(row)
            except ValueError:
                print(f"Could not convert line to floats: {line}")

# Convert data to numpy array. shape: N x nvar 
# traspose the matrix if it is nvar*N 
if np.array(data_in_file).shape[1] > np.array(data_in_file).shape[0]:
    time_series = np.transpose(np.array(data_in_file))
else: 
    time_series = np.array(data_in_file)  

if time_series.shape[1] == 37: 
    time_series = np.delete(time_series, 0, 1) # delete first column = time points

print("shape of time series: ", np.shape(time_series))

r = 20
s = 36

# Call the function
results = information_flow_subspace(time_series, r-1, s-1, np_val=1, n_iter=1000, alpha=0.01)

# Print results
print("TAB:", results["TAB"])
print("TBA:", results["TBA"])
print("Error TAB:", results["error_TAB"])
print("Error TBA:", results["error_TBA"])
print("Significant TAB:", results["significance_TAB (bool, Z, p-value)"])
print("Significant TBA:", results["significance_TBA (bool, Z, p-value)"])

print("elapsed time: ", time.time() - start)

# Extract file name and change extension to .csv
output_file = os.path.splitext(file_name)[0] + ".csv"

# Save results to CSV
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow(["Metric", "Value"])
    
    # Write results
    writer.writerow(["TAB", results["TAB"]])
    writer.writerow(["TBA", results["TBA"]])
    writer.writerow(["Error TAB", results["error_TAB"]])
    writer.writerow(["Error TBA", results["error_TBA"]])
    writer.writerow(["Significant TAB", results["significance_TAB (bool, Z, p-value)"]])
    writer.writerow(["Significant TBA", results["significance_TBA (bool, Z, p-value)"]])

print(f"Results saved to {output_file}")
print("elapsed time: ", time.time() - start)