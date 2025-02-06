"""
in this file I will store functions I can call in other files to perform the analyses 
without needing to re-write them every time. 
"""

import numpy as np 
import csv 
from scipy import stats

import sys
sys.path.insert(0, '/home/chiaraz/Liang_Index_climdyn')
from function_liang_nvar import compute_liang_nvar

# Function to convert days to data points
def convert_days_to_points(days):
    points_per_day = 1 / 11
    return int(days * points_per_day)


# This function averages out atmopsheric time series
# valid for 1D fourier times series and for real fields 
def average_time_series(data, window_size_days, apply_averaging):
    if not apply_averaging:
        return data  # Return original data if averaging is not applied

    nvar, ntime = data.shape  # Correct shape extraction
    print(data.shape) # expected 37 * N 
    window_size = convert_days_to_points(window_size_days)

    if window_size > ntime:
        raise ValueError("Window size is larger than the time series length.")

    # Calculate the number of averages that can be computed
    # the last window_size -1 points are lost 
    new_length = ntime - window_size + 1
    print("new time series length: ", new_length)

    averaged_data = np.empty((nvar, new_length))

    # Perform running mean
    for i in range(new_length):
        start = i
        end = start + window_size
        # averages out atmopsheric time series 
        averaged_data[1:21, i] = data[1:21:, start:end].mean(axis=1)

    # time and oceanic time series are left unchanged
    averaged_data[21:,:] = data[21:, :new_length]
    return averaged_data
    print("Succesfully applied averaging to data.")


# LAG function for diagnostics (coming straight from the code)
def introduce_lag_diagnostics(data1, data2, days_of_delay, apply_lag):
    """
    Introduces a lag between two 3D datasets in the format [time, x, y] and aligns them after applying the lag.

    Parameters:
        data1 (np.ndarray): The first dataset (time, x, y).
        data2 (np.ndarray): The second dataset (time, x, y).
        months_of_delay (float): Delay in months (positive = data2 lags backward, negative = data1 lags backward).
        points_per_month (float): Conversion factor for time points per month.
        apply_lag (bool): Whether to apply the lag.

    Returns:
        tuple: A tuple containing the two datasets after applying the lag and trimming.
    """
    if not apply_lag or days_of_delay == 0:
        return data1, data2  # Return the original data if no lag is applied

    # Convert months of delay to time points
    lag_points = convert_days_to_points(days_of_delay)

    # Shape of the data: (time, x, y)
    time, x, y = data1.shape

    if lag_points > 0:
        # data2 lags behind data1
        trimmed_data1 = data1[:-lag_points, :, :]  # Trim the first dataset
        trimmed_data2 = data2[lag_points:, :, :]  # Trim the second dataset
    else:
        # data1 lags behind data2
        lag_points = abs(lag_points)
        trimmed_data1 = data1[lag_points:, :, :]  # Trim the first dataset
        trimmed_data2 = data2[:-lag_points, :, :]  # Trim the second dataset

    # Align the time dimensions after trimming
    min_time = min(trimmed_data1.shape[0], trimmed_data2.shape[0])
    trimmed_data1 = trimmed_data1[:min_time, :, :]
    trimmed_data2 = trimmed_data2[:min_time, :, :]

    print(f"Lag applied: {days_of_delay} days ({lag_points} time points)")
    return trimmed_data1, trimmed_data2

    print("successfully introduced lag between 3D time series")


# Averaging function for functions of shape (time, x, y)
def average_time_series_3D(data, window_size_days, apply_averaging):
    """
    Averages the data over a manually set number of time points along the time axis.

    Parameters:
        data (np.ndarray): The input time-series array with shape (time, x, y).
        window_size (int): Number of time points to average over.
        apply_averaging (bool): Whether to apply averaging.

    Returns:
        np.ndarray: The averaged time-series array with shape (new_time, x, y),
                    or the original array if averaging is not applied.
    """
    if not apply_averaging:
        return data  # Return the original data if averaging is not applied

    # Extract dimensions
    time, x, y = data.shape
    window_size = convert_days_to_points(window_size_days)
    new_time = time // window_size

    # Initialize an array to store the averaged results
    averaged_data = np.empty((new_time, x, y))
    
    # Perform the averaging along the time dimension
    for i in range(new_time):
        start = i * window_size
        end = start + window_size
        averaged_data[i, :, :] = data[start:end, :, :].mean(axis=0)

    return averaged_data

    print("succesfully averaged 3D time series")

# Function to introduce LAG between atmospheric and oceanic variables in fourier space. 
# all atmospheric functions are lagged by the same amount, same for ocean 
def introduce_lag_fourier(data, days_of_delay, apply_delay):
    if not apply_delay:
        return data  # No lag applied
    
    lag_points = convert_days_to_points(days_of_delay)
    print("Data was lagged of", lag_points, "data points")
    n_atmospheric = 20  # First 20 variables are atmospheric
    n_oceanic = 16      # Last 16 variables are oceanic

    atmosphere = data[:n_atmospheric, :]
    ocean = data[n_atmospheric:, :]

    if lag_points > 0:
        # Ocean lags behind atmosphere
        trimmed_atmosphere = atmosphere[:, :-lag_points]
        trimmed_ocean = ocean[:, lag_points:]
    else:
        # Atmosphere lags behind ocean
        lag_points = abs(lag_points)
        trimmed_atmosphere = atmosphere[:, lag_points:]
        trimmed_ocean = ocean[:, :-lag_points]

    min_time = min(trimmed_atmosphere.shape[1], trimmed_ocean.shape[1])
    return np.vstack((trimmed_atmosphere[:, :min_time], trimmed_ocean[:, :min_time]))

    print("Lag introduced succesfully.")

# Function to compute Liang's metrics and save results to CSV
def compute_and_save_liang_results(time_series, subset_variables, output_filename):
    """
    time series : time series input of liang's function 
    variables : subset of variables in the time series to consider 
    output_filename : the name of the file where results are stored    
    """
    nvar_results = np.array(compute_liang_nvar(time_series, 1, 1000))

    num_vars = len(subset_variables)
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


# add annotations on the plot of Liang's result
def add_annotations(ax, data, mask):
    """
    Add annotations to the plot, but only for values that are not masked.
    
    Parameters:
    - ax: The axis to which annotations will be added.
    - data: The data array being plotted.
    - mask: A boolean mask array where True indicates a value should not be annotated.
    """
    for (i, j), val in np.ndenumerate(data):
        if not mask[i, j]:  # Only annotate if the value is not masked
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=7)


def is_extreme_value(values, threshold=0.999):
    """
    Determines if values are in the top x% of extreme high values.

    Args:
        values (matrix-like): The set of values to evaluate.
        threshold (float): The percentile threshold (default is 99% for top 1% values).

    Returns:
        array: Boolean mask indicating whether each value is in the top 1%.
    """
    values = np.array(values)
    
    if values.size == 0:  # Handle empty input
        return np.array([], dtype=bool)
    
    percentile_threshold = np.percentile(values, threshold * 100)  # 99th percentile
    print("threshold:", percentile_threshold)

    # Check if the threshold is NaN, infinite, or too large
    if not np.isfinite(percentile_threshold):
        print("Warning: Computed percentile is not a valid number. Returning all False mask.")
        return np.zeros_like(values, dtype=bool)
    
    for value in values:
        if value >= percentile_threshold:
            print("exceeding value:", value)

    return values >= percentile_threshold  # Boolean mask for extreme high values

# definitely need to do this better. it's not about percentiles it's about extremes. s

import numpy as np

def apply_masking(data, threshold_for_extremes, source_column="Source", target_column="Target", 
                  tau_column="Tau", tau_error_column="Error_Tau",
                  r_column="R", r_error_column="Error_R", use_masking=True):
    """
    Function to set not-singificant and extreme values to 0 in Tau and R matrices.
    """
    source_vars = sorted(data[source_column].str.replace("Var", "").astype(int).unique())
    target_vars = sorted(data[target_column].str.replace("Var", "").astype(int).unique())

    num_sources = len(source_vars)
    num_targets = len(target_vars)

    # Initialize matrices with actual values (not masked)
    tau_matrix = np.zeros((num_sources, num_targets))
    r_matrix = np.zeros((num_sources, num_targets))

    source_to_idx = {v: i for i, v in enumerate(source_vars)}
    target_to_idx = {v: i for i, v in enumerate(target_vars)}

    tau_values = []
    r_values = []

    # Fill matrices with real values
    for _, row in data.iterrows():
        source = int(row[source_column].replace("Var", ""))
        target = int(row[target_column].replace("Var", ""))
        
        tau_value = row[tau_column]
        r_value = row[r_column]
        error_tau = row[tau_error_column]
        error_r = row[r_error_column]

        if source in source_to_idx and target in target_to_idx:
            s_idx = source_to_idx[source]
            t_idx = target_to_idx[target]

            tau_matrix[s_idx, t_idx] = tau_value
            r_matrix[s_idx, t_idx] = r_value

            tau_values.append(tau_value)
            r_values.append(r_value)

    # Identify extreme values
    tau_extreme_mask = is_extreme_value(tau_values, threshold_for_extremes)  # Top x% are marked True

    # Convert lists to numpy arrays for indexing
    tau_values = np.array(tau_values)
    r_values = np.array(r_values)

    # Set insignificant and extreme values to zero
    for i, (_, row) in enumerate(data.iterrows()):
        source = int(row[source_column].replace("Var", ""))
        target = int(row[target_column].replace("Var", ""))

        if source in source_to_idx and target in target_to_idx:
            s_idx = source_to_idx[source]
            t_idx = target_to_idx[target]

            # Check for non-significant values (error bars crossing zero)
            tau_lower = row[tau_column] - row[tau_error_column]
            tau_upper = row[tau_column] + row[tau_error_column]
            if tau_lower <= 0 <= tau_upper or tau_extreme_mask[i]:  # Also zero extreme values
                #print("not significant tau value: ", tau_lower)
                tau_matrix[s_idx, t_idx] = 0  

            r_lower = row[r_column] - row[r_error_column]
            r_upper = row[r_column] + row[r_error_column]
            if r_lower <= 0 <= r_upper:
                #print("not significant r value: ", r_lower)
                r_matrix[s_idx, t_idx] = 0  
    
    return tau_matrix, r_matrix, source_vars, target_vars


