import numpy as np
import csv
import time
import os 
import matplotlib.pyplot as plt

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
file_path = 'Normalized_Climate_anomalies.csv'
data_in_file = []

with open("Normalized_Climate_anomalies.csv", 'r') as file:
    reader = csv.reader(file)
    header = next(reader)   # ['EPAC','CPAC','NAT','WPAC']
    data = []
    for row in reader:
        try:
            values = [float(x) for x in row]
            data.append(values)
        except ValueError:
            print(f"Skipping bad line: {row}")

data = np.array(data)  
print("data shape: ", np.shape(data)) # shape (N, 4)

# Convert data to numpy array and transpose ==> (nvar, N) matrix
time_series = np.transpose(data)
print("shape of original time series: ", np.shape(time_series))


if np.any(np.isnan(time_series)) or np.any(np.isinf(time_series)):
    raise ValueError("Time series contains NaN or Inf values!")
time_series[np.isnan(time_series)] = 0
time_series[np.isinf(time_series)] = 0

# Compute Liang's function
nvar_results = np.array(compute_liang_nvar(time_series, 1, 1000))

# Prepare arrays for results
num_vars = 4
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

results_folder = "."
os.makedirs(results_folder, exist_ok=True)

# Save results to a CSV file
output_file = os.path.join(results_folder, "liang_results.csv")
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

print(f"{output_file}")
print("Execution time:", time.time() - start)
