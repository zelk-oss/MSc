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

results_folder = "results_fast_slow"
os.makedirs(results_folder, exist_ok=True)
def compute_save_liang(data, output_file_name): 
    # Compute Liang's function
    nvar_results = np.array(compute_liang_nvar(data, 1, 1000))

    # Prepare arrays for results
    num_vars = 36
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

    # Save results to a CSV file
    output_file = os.path.join(results_folder, output_file_name)
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

    print(f"Results saved to fast_slow_dynamics/results/fast_slow/{output_file}")
    print("Execution time:", time.time() - start)

def plot_time_series(data, series_index, title="Time Series Plot"):
    """
    Plot a selected time series.
    :param data: numpy array of shape (36, N)
    :param series_index: index of the series to plot (0 to 35)
    :param title: Title of the plot
    """
    if series_index < 0 or series_index >= data.shape[0]:
        print("Invalid series index. Must be between 0 and 35.")
        return
    
    plt.figure(figsize=(10, 5))
    plt.plot(data[series_index, :], label=f"Variable {series_index+1}")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

#######################
####### FOR TE ########
#######################
# save to file a chunk of this file for the TE analysis 
def keep_middle_window_TE(input_array, output_file):
    # Ensure the output folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Get total points (number of columns)
    total_points = input_array.shape[1]
    print(f"Total points: {total_points}")

    # Parameters
    window_size = 10000  # Number of points to keep
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

def keep_middle_window_LKIF(input_array):
    # Get total points (number of columns)
    total_points = input_array.shape[1]
    print(f"Total points: {total_points}")

    # Parameters
    window_size = 10000  # Number of points to keep
    middle_index = total_points // 2
    half_window = window_size // 2

    # Select the range around the middle index
    start_index = max(0, middle_index - half_window)
    end_index = min(total_points, middle_index + half_window)
    selected_lines = input_array[:, start_index:end_index]
    
    return selected_lines


# check for invalied values
def remove_nan(time_series):
    if np.any(np.isnan(time_series)) or np.any(np.isinf(time_series)):
        raise ValueError("Time series contains NaN or Inf values!")
    time_series[np.isnan(time_series)] = 0
    time_series[np.isinf(time_series)] = 0


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
time_series = np.transpose(np.array(data_in_file, dtype=np.float64))
select_vars = list(range(1, 37))
time_series = time_series[select_vars,:]
print("shape of original time series: ", np.shape(time_series))

# Assuming `time_series` has shape (36, N) (35 variables + 1 reference)
fast_threshold = 0.033
slow_threshold = 0.030

# Reference time series
reference_series = time_series[0]

# Create masks based on the reference series
fast_mask = reference_series > fast_threshold
slow_mask = reference_series < slow_threshold


"""
# Apply masks to all time series
fast_time_series = np.where(fast_mask, time_series, 0)
slow_time_series = np.where(slow_mask, time_series, 0)
# keep only central window 
fast_time_series = keep_middle_window_LKIF(fast_time_series)
slow_time_series = keep_middle_window_LKIF(slow_time_series)

# Keeping a chunk of data for TE analysis
keep_middle_window_TE(fast_time_series, "/home/chiaraz/data_thesis/data_1e5points_1000ws/window_for_TE/fast_slow_dynamics/fast_TE_timeseries_weak_0avg")
keep_middle_window_TE(slow_time_series, "/home/chiaraz/data_thesis/data_1e5points_1000ws/window_for_TE/fast_slow_dynamics/slow_TE_timeseries_weak_0avg")


# liang and output for the fast time series 
compute_save_liang(fast_time_series, "results_fast_weak_0avg.csv")
# liang and output for the slow time series 
compute_save_liang(slow_time_series, "results_slow_weak_0avg.csv")
"""

##############################
##### removing points ########
##############################
# Applichiamo la maschera, eliminando i punti corrispondenti in tutte le 36 variabili
Rfast_time_series = time_series[:, fast_mask]
Rslow_time_series = time_series[:, slow_mask]

Rfast_time_series = keep_middle_window_LKIF(Rfast_time_series)
Rslow_time_series = keep_middle_window_LKIF(Rslow_time_series)

# Controlliamo la nuova forma delle serie temporali
print("Nuova forma time series FAST con removal:", Rfast_time_series.shape)
print("Nuova forma time series SLOW con removal :", Rslow_time_series.shape)

# Keeping a chunk of data for TE analysis
keep_middle_window_TE(Rfast_time_series, "/home/chiaraz/data_thesis/data_1e5points_1000ws/window_for_TE/fast_slow_dynamics/REMOVEDfast_TE_timeseries_weak_0avg")
keep_middle_window_TE(Rslow_time_series, "/home/chiaraz/data_thesis/data_1e5points_1000ws/window_for_TE/fast_slow_dynamics/REMOVEDslow_TE_timeseries_weak_0avg")

# liang and output for the fast time series 
compute_save_liang(Rfast_time_series, "REMOVEDresults_fast_weak_0avg.csv")
# liang and output for the slow time series
compute_save_liang(Rslow_time_series, "REMOVEDresults_slow_weak_0avg.csv")


"""
plot_time_series(time_series, 0, title="Original Time Series")
plot_time_series(slow_time_series, 0, title="Slow Dynamics Time Series")
plot_time_series(slow_time_series, 11, title="Slow Dynamics Time Series with removed data")
plot_time_series(fast_time_series, 0, title="Fast Dynamics Time Series")
plot_time_series(fast_time_series, 11, title="Fast Dynamics Time Series with removed data")
"""