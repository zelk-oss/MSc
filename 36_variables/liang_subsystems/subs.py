"""
This script computes the subsystems LKIF calling the function from file
"compute_liang_subsystems". Also implements p-value analysis
"""

import numpy as np
import csv
import time

# add things for path if you move files around 
from liang_subsystems.compute_liang_subsystems import information_flow_subspace

# Start timer
start = time.time()

# Load time series data
accelerate = 1
file_path = '../myqgs/data_1e5points_1000ws/evol_fields_1e-8.dat'
data_in_file = []

with open(file_path, 'r') as file:
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
time_series = np.delete(np.array(data_in_file), 0, 1) # delete first column = time points 

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
print("Significant TAB:", results["significance_TAB"])
print("Significant TBA:", results["significance_TBA"])

print("elapsed time: ", time.time() - start)