import numpy as np
import csv
import time
import os 

# Import Liang's bivariate formula
import sys
sys.path.insert(0, '/home/chiaraz/Liang_Index_climdyn')
from function_liang_nvar import compute_liang_nvar

sys.path.insert(0, '/home/chiaraz/thesis')
from functions_for_maooam import average_time_series

# Start timer
start = time.time()

# Load time series data
accelerate = 1
file_path = '../../../data_thesis/data_1e5points_1000ws/evol_fields_1e-8.dat'
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

# Convert data to numpy array and transpose ==> (nvar, N) matrix
time_series = np.transpose(np.array(data_in_file))
print("shape of original time series: ", np.shape(time_series))

# Apply averaging
window_size_10yr= 3652.4
time_series_10yr_averaging = average_time_series(time_series, window_size_10yr, True)


# Select variables
select_vars = list(range(1, 37))
select_time_series_10yr_averaging = time_series_10yr_averaging[select_vars, :]
print("length of the new time series: ", np.shape(select_time_series_10yr_averaging)) # 36 variables left 

"""
this is for TE alysis: saving some data
"""

"""
# save to file a chunk of this file for the TE analysis 
def keep_middle_window(input_array, output_file):
    # Ensure the output folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Get total points (number of columns)
    total_points = input_array.shape[1]
    print(f"Total points: {total_points}")

    # Parameters
    window_size = 12500  # Number of points to keep
    middle_index = total_points // 2
    half_window = window_size // 2

    # Select the range around the middle index
    start_index = max(0, middle_index - half_window)
    end_index = min(total_points, middle_index + half_window)
    selected_lines = input_array[:, start_index:end_index]

    # Write the selected lines to the output file
    with open(output_file, 'w') as file:
        for row in selected_lines:
            # Convert each row to a tab-separated string and write it to the file
            file.write('\t'.join(map(str, row)) + '\n')

    print(f"Successfully kept a window of {window_size} points around the middle. Output saved to {output_file}.")

# Keeping a chunk of data for TE analysis (very slow)
keep_middle_window(select_time_series_10yr_averaging, "/home/chiaraz/data_thesis/data_1e5points_1000ws/window_for_TE/avg_atmo/100yr_strong_largewindow")
"""

# Compute Liang's function
nvar_results = np.array(compute_liang_nvar(select_time_series_10yr_averaging, 1, 3))

# Prepare arrays for results
num_vars = len(select_vars)
tau_nvar = np.empty((num_vars, num_vars))
error_tau_nvar = np.empty((num_vars, num_vars))
r_nvar = np.empty((num_vars, num_vars))
error_r_nvar = np.empty((num_vars, num_vars))
info_flow = np.empty((num_vars, num_vars))
error_info_flow = np.empty((num_vars, num_vars))

for i in range(num_vars):
    for j in range(num_vars):
        info_flow[i, j] = nvar_results[0, i, j]
        tau_nvar[i, j] = nvar_results[1, i, j]
        r_nvar[i, j] = nvar_results[2, i, j]
        error_tau_nvar[i, j] = nvar_results[4, i, j]
        error_r_nvar[i, j] = nvar_results[5, i, j]
        error_info_flow[i, j] = nvar_results[3, i, j]

results_folder = "results_averaging_atmosphere"
os.makedirs(results_folder, exist_ok=True)

# Save results to a CSV file
output_file = os.path.join(results_folder, "test.csv")
csv_headers = [
    "Source", "Target", "InfoFlow", "Error_InfoFlow", "Tau", "Error_Tau", "R", "Error_R"
]

csv_rows = []
for i in range(num_vars):
    for j in range(num_vars):
        row_data = [
            f"Var{i + 1}",  # Source
            f"Var{j + 1}",  # Target
            info_flow[i, j],
            error_info_flow[i, j],
            tau_nvar[i, j],
            error_tau_nvar[i, j],
            r_nvar[i, j],
            error_r_nvar[i, j],
        ]
        csv_rows.append(row_data)

with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(csv_headers)
    writer.writerows(csv_rows)

print(f"Results saved to results_averaging_atmosphere/{output_file}")
print("Execution time:", time.time() - start)
