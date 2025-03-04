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
from functions_for_maooam import introduce_lag_fourier

# Function to compute Liang's metrics and save results to CSV
def compute_and_save_liang_results(time_series, variables, output_filename):
    """
    time series : time series input of liang's function 
    variables : subset of variables in the time series to consider 
    output_filename : the name of the file where results are stored    
    """
    nvar_results = np.array(compute_liang_nvar(time_series, 1, 100)) # 1000 bootstrap iterations 

    num_vars = len(variables)
    csv_headers = [
        "Source", "Target", "InfoFlow", "Error_InfoFlow", "Tau", "Error_Tau", "R", "Error_R"
    ]
    csv_rows = []

    for i in range(num_vars):
        for j in range(num_vars):
            row_data = [
                f"Var{i + 1}",  # Source
                f"Var{j + 1}",  # Target
                nvar_results[0, i, j],
                nvar_results[3, i, j],
                abs(nvar_results[1, i, j]),
                nvar_results[4, i, j],
                nvar_results[2, i, j],
                nvar_results[5, i, j],
            ]
            csv_rows.append(row_data)

    with open(output_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)
        writer.writerows(csv_rows)

    print(f"Results saved to {output_filename}")


# save to file a chunk of this file for the TE analysis 
def keep_middle_window(input_array, output_file):
    # Ensure the output folder exists
  
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


# Main execution
if __name__ == "__main__":
    start = time.time()
    
    accelerate = 1
    
    file_path = '../../../data_thesis/data_1e5points_1000ws/evol_fields_1_1e-7.dat'

    data_in_file = []

    # Load data file
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

    # Convert and transpose data
    time_series = np.transpose(np.array(data_in_file))  # result: (nvar * N)
    print(time_series[0, :])  # prints the data points 
    # apply averaging if needed 

    # select variables 
    select_vars = list(range(1,37))  # select columns from the second to the end 
    select_time_series = time_series[select_vars, :] # ignore time column 

    # Define lags in days
    lags = [100, 3000]  # Lags in days

    # Process each lag and save results
    results_folder = "liang_lagging_results"
    os.makedirs(results_folder, exist_ok=True)

    for i, days in enumerate(lags):
        lagged_data = introduce_lag_fourier(select_time_series, days, True)
        keep_middle_window(lagged_data, f"window_lag_strong_{lags[i]}days")
        print("shape of lagged data", np.shape(lagged_data))
        #output_filename = os.path.join(results_folder, f"results_data11days_WEAK_lag_{lags[i]}days.csv")
        #compute_and_save_liang_results(lagged_data, select_vars, output_filename)

    print("Execution time =", time.time() - start)
