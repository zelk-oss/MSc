"""
This script computes the subsystems LKIF calling the function from file
"compute_liang_subsystems". Also implements p-value analysis but not to be trusted completely 
This version of the file handles a batch of files with different lags or averaged applied, 
realized to plot the |LKIF| as a function of lagging or averaging 
"""
import numpy as np
import csv
import time
import os

# Add things for path if you move files around
from compute_liang_subsystems import information_flow_subspace

# Start timer
start = time.time()

# Define the input and output directories
file_path = '/home/chiaraz/data_thesis/data_1e5points_1000ws/window_for_TE/batch_log_lag/'
output_folder = "batch_log_lag/"

#days = 20   # for linspace data 
vector_days = np.logspace(1.0, 4.2, 100) # 200 values logarithmically spaced from 1e1 to 1e4.2 = 40 years 

# Open the CSV file for writing all results
output_file = output_folder + "log_results_weak_BIGERROR.csv"

# Prepare the header for the CSV file (first row)
header = ["File", "TAB", "TBA", "Error TAB", "Error TBA", "Significant TAB", "Significant TBA"]

# Write header to the CSV file (create file if it doesn't exist)
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

# Loop through the files
#for i in range(1, 101): this is the option for linspace data from 20 days average to 1000 days average 
    # Define the file name dynamically
for i in range(0,101):
    file_name = f'{i}w'
    file = file_path + file_name
    data_in_file = []

    # Read the data from the file
    with open(file, 'r') as f:
        for index, line in enumerate(f):
            if index % 1 == 0:  # adjust this as needed for acceleration
                try:
                    row = [float(value) for value in line.split()]
                    if len(row) < 37:
                        print(f"Line has insufficient data: {line.strip()}")
                        continue
                    data_in_file.append(row)
                except ValueError:
                    print(f"Could not convert line to floats: {line}")

    # Convert data to numpy array and transpose if needed
    if np.array(data_in_file).shape[1] > np.array(data_in_file).shape[0]:
        time_series = np.transpose(np.array(data_in_file))
    else: 
        time_series = np.array(data_in_file)

    # If shape has 37 variables, delete the first column (time points)
    if time_series.shape[1] == 37:
        time_series = np.delete(time_series, 0, 1)

    print("Shape of time series:", np.shape(time_series))

    # Parameters for the function call
    r = 20
    s = 36

    # Call the function to compute results
    results = information_flow_subspace(time_series, r-1, s-1, np_val=1, n_iter=50, alpha=0.05)

    # Write the results to the combined CSV file
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Create the row with the file label and the corresponding results
        row = [
            file_name,
            results["TAB"],
            results["TBA"],
            results["error_TAB"],
            results["error_TBA"],
            results["significance_TAB (bool, Z, p-value)"],
            results["significance_TBA (bool, Z, p-value)"]
        ]
        
        # Write the row to the CSV file
        writer.writerow(row)

    print(f"Results for {file_name} saved to {output_file}")
    
    # Increment the day value for the next file
    #days += 10

print("Elapsed time: ", time.time() - start)
