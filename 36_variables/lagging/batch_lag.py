import numpy as np
import csv
import time
import os

"""
Extract time series from the strong coupling and the weak coupling files. 
Iteratively apply the lagging function, and optionally the averaging function, to time series.
Save windows of data (10000 points = 300 years) to perform both TE and LKIF multivariate analysis 
Save these in the data_thesis folder in a new subdirectory "batch_log_lag" 
"""

# Import Liang's bivariate formula
import sys
sys.path.insert(0, '/home/chiaraz/Liang_Index_climdyn')
from function_liang_nvar import compute_liang_nvar

sys.path.insert(0, '/home/chiaraz/thesis')
from functions_for_maooam import average_time_series
from functions_for_maooam import introduce_lag_fourier

# checking if lag functions as expected 
import matplotlib.pyplot as plt

def plot_lag_examples(i, vector_days_i, original, lagged, nlagged, var_a=1, var_o=21):
    total_points = original.shape[1]
    zoom_window = total_points // 10
    start_index = total_points // 2 - zoom_window // 2
    end_index = start_index + zoom_window
    t = np.arange(zoom_window)

    # 1. Plot originale (nessun lag)
    plt.figure(figsize=(10, 4))
    plt.plot(t, original[var_a, start_index:end_index], label='Atmosfera (originale)', color='blue')
    plt.plot(t, original[var_o, start_index:end_index], label='Oceano (originale)', color='red')
    plt.title(f'[i={i}] Nessun lag (originale)')
    plt.xlabel('Time steps (zoomed)')
    plt.ylabel('Valore')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Plot con lag positivo
    plt.figure(figsize=(10, 4))
    plt.plot(t, lagged[var_a, start_index:end_index], label='Atmosfera (lag +)', linestyle='--', color='blue')
    plt.plot(t, lagged[var_o, start_index:end_index], label='Oceano (lag +)', linestyle='--', color='red')
    plt.title(f'[i={i}] Lag positivo: +{vector_days_i:.1f} giorni')
    plt.xlabel('Time steps (zoomed)')
    plt.ylabel('Valore')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Plot con lag negativo
    plt.figure(figsize=(10, 4))
    plt.plot(t, nlagged[var_a, start_index:end_index], label='Atmosfera (lag -)', linestyle=':', color='blue')
    plt.plot(t, nlagged[var_o, start_index:end_index], label='Oceano (lag -)', linestyle=':', color='red')
    plt.title(f'[i={i}] Lag negativo: -{vector_days_i:.1f} giorni')
    plt.xlabel('Time steps (zoomed)')
    plt.ylabel('Valore')
    plt.legend()
    plt.tight_layout()
    plt.show()

"""
questo per il batch non ci serve 
# Function to compute Liang's metrics and save results to CSV
def compute_and_save_liang_results(time_series, variables, output_filename):

    time series : time series input of liang's function 
    variables : subset of variables in the time series to consider 
    output_filename : the name of the file where results are stored    

    nvar_results = np.array(compute_liang_nvar(time_series, 1, 1000)) # 1000 bootstrap iterations 

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
"""

# keep window of data 
def keep_middle_window(select_time_series):
    # Get total points (number of columns)
    total_points = select_time_series.shape[1]
    #print(f"Total points: {total_points}")

    # Parameters
    window_size = 10000 # Number of points to keep
    middle_index = total_points // 2
    half_window = window_size // 2

    # Select the range around the middle index
    start_index = max(0, middle_index - half_window)
    end_index = min(total_points, middle_index + half_window)
    selected_window = select_time_series[:, start_index:end_index]
    return selected_window

# Main execution
if __name__ == "__main__":
    start = time.time()
    
    accelerate = 1
    
    file_path = '../../../data_thesis/data_1e5points_1000ws/evol_fields_1e-8.dat'

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
    print("shape of time series", np.shape(time_series))
    #print(time_series[0, :])  # prints the data points 
    # apply averaging if needed 

    # select variables 
    select_vars = list(range(1,37))  # select columns from the second to the end 
    select_time_series = time_series[select_vars, :] # ignore time column 
    print("shape of selected time series", np.shape(select_time_series))

    # Vcetor of lags in days
    vector_days = np.logspace(1.0, 4.2, 101, True) # 100 points logarithmically spaced from 

    # Process each lag and save results
    results_folder = "/home/chiaraz/data_thesis/data_1e5points_1000ws/window_for_TE/batch_log_lag/"
    os.makedirs(results_folder, exist_ok=True)

    plot_indices = np.linspace(0, 100, 5, dtype=int)

    # first number has to be higher than 2 
    for i in range(0,101):
        print(i)
        if vector_days[i] < 11: 
            lagged_data = introduce_lag_fourier(select_time_series, vector_days[i], False)
            nlagged_data = introduce_lag_fourier(select_time_series, vector_days[i], False)
        else: 
            lagged_data = introduce_lag_fourier(select_time_series, vector_days[i]*(1), True)
            # also negative lagged data, atmosphere falling behind 
            nlagged_data = introduce_lag_fourier(select_time_series, vector_days[i]* (-1), True)
        
        print("number of days of lag ", vector_days[i])
        print("shape lagged_data", np.shape(lagged_data))
        print("shape n_lagged_data", np.shape(nlagged_data))

        window_lagged_data = keep_middle_window(lagged_data)
        print(f"shape of kept window: {np.shape(window_lagged_data)}")
        window_nlagged_data = keep_middle_window(nlagged_data)
        print("shape of kept window: ",np.shape(window_nlagged_data))

        # check lags visually 
        if i in plot_indices:
            plot_lag_examples(i, vector_days[i], select_time_series, window_lagged_data, window_nlagged_data)

        """
        output_filename = os.path.join(results_folder, f"{i}w")
        # negative lag filename 
        noutput_filename = os.path.join(results_folder, f"neg{i}w")

        # Write the selected lines to the output file
        with open(output_filename, 'w') as file:
            for row in window_lagged_data:
                # Convert each row to a tab-separated string and write it to the file
                file.write('\t'.join(map(str, row)) + '\n')
        print(f"Successfully kept a window of {window_lagged_data.shape[1]} points around the middle. Output saved to {output_filename}.")

        # also for negative lag 
        with open(noutput_filename, 'w') as file:
            for row in window_nlagged_data:
                # Convert each row to a tab-separated string and write it to the file
                file.write('\t'.join(map(str, row)) + '\n')
        print(f"Successfully kept a window of {window_nlagged_data.shape[1]} points around the middle. Output saved to {noutput_filename}.")
        """
    print("Execution time = ", time.time() - start)
