# script to compute and display the information flow vs the correlation coefficient 
# applied to the time series of the maooam model 

import numpy as np 
import matplotlib.pyplot as plt
import csv 
import time

""" space for functions """

def convert_days_to_points(days):
    """
    Converts months into time points based on a given rate.

    Parameters:
        months (float): The number of months to convert.
        points_per_month (float): Conversion factor for time points per month.

    Returns:
        int: Equivalent number of time points.
        every data point = 0.11 days
    """
    points_per_day = 1/0.11
    return int(days * (points_per_day))


# optional function to average over some time window 
def average_time_series(data, window_size_days, apply_averaging):
    """
    Averages the data over a manually set number of time points.

    Parameters:
        data (np.ndarray): The input time_series array (nvar x N matrix).
        window_size (int): Number of months to average over.
        apply_averaging (bool): Whether to apply averaging.

    Returns:
        np.ndarray: The averaged time_series array (or original if not applied).
    """
    if not apply_averaging:
        return data  # Return the original data if averaging is not applied

    # Compute the new length after averaging
    nvar, ntime = data.shape
    # obtain window size in data points 
    window_size = convert_days_to_points(window_size_days)
    new_length = ntime // window_size

    # Initialize an array to store averaged results
    averaged_data = np.empty((nvar, new_length))

    # Perform the averaging
    for i in range(new_length):
        start = i * window_size
        end = start + window_size
        averaged_data[:, i] = data[:, start:end].mean(axis=1)

    return averaged_data

# optional function to introduce a lag between ocean and atmospheric functions 
def introduce_lag(data, days_of_delay, apply_lag):
    """
    Introduces a lag between atmospheric and oceanic variables in the dataset.
    Trims data to align all columns after introducing the lag.

    Parameters:
        data (np.ndarray): The input dataset (nvar x N matrix).
        months_of_delay (float): Delay in months (positive = ocean lags, negative = atmosphere lags).
        points_per_month (float): Conversion factor for time points per month.
        apply_lag (bool): Whether to apply the lag.

    Returns:
        np.ndarray: The lagged dataset with aligned columns.
    """
    if not apply_lag or days_of_delay == 0:
        return data  # Return the original data if no lag is applied

    # Convert months of delay to time points
    lag_points = convert_days_to_points(days_of_delay)

    # Number of variables
    nvar, ntime = data.shape
    n_atmospheric = 20  # First 20 variables are atmospheric
    n_oceanic = 16      # Last 16 variables are oceanic

    # Split data into atmospheric and oceanic variables
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

    # Align the time dimensions after trimming
    min_time = min(trimmed_atmosphere.shape[1], trimmed_ocean.shape[1])
    trimmed_atmosphere = trimmed_atmosphere[:, :min_time]
    trimmed_ocean = trimmed_ocean[:, :min_time]

    # Combine atmospheric and oceanic variables back into a single dataset
    lagged_data = np.vstack((trimmed_atmosphere, trimmed_ocean))

    print(f"Lag applied: {days_of_delay} days ({lag_points} time points)")
    return lagged_data

# optional function to add value of IF/correlation in the plots 
def add_annotations(ax, data):
    for (i, j), val in np.ndenumerate(data):
        text_val = f"{val:.2f}".replace('-', '− ')  # Replace hyphen with en-dash and add a space
        ax.text(j, i, text_val, ha='center', va='center', fontsize=7)

"""end of functions """

start = time.time()

# importing Liang's bivariate formula from another folder 
import sys 
sys.path.insert(0, '/home/chiaraz/Liang_Index_climdyn')
from function_liang_nvar import compute_liang_nvar

# Initialize an empty 36x36 matrix to store the results of compute_liang
result_matrix = np.empty((36, 36), dtype=object)
data_in_file = []

# opening the time series file in /qgs directory  
accelerate = 10
file_path = '../qgs/evol_fields_1_1e-7.dat'
with open(file_path, 'r') as file:
    for index, line in enumerate(file):
        if index % accelerate == 0:
            # handling potential invalid data 
            try:
                # turns lines in the file in tuples of 37 elements in the maooam case
                row = [float(value) for value in line.split()]
                if len(row) < 37: 
                    print("Line has insufficient data: {line.strip()}")
                    continue 
                # here we can do stuff with the data 
                data_in_file.append(row)

            except ValueError:
                print(f"Could not convert line to floats: {line}")

# Convert the data list to a numpy 2D array (shape: N x 37)
time_series = np.transpose(np.array(data_in_file)) # nvar x N matrix 

# apply the function to a subset of variables 
select_vars = list(range(1, 37))



# apply different averagings. compute fuction and draw plots for all types 

# experiment 1 
select_time_series_0_averaging = time_series[select_vars,:]
print(select_time_series_0_averaging.shape)

# experiment 2 : averaging over 100 days 
window_size_10 = 10
time_series_10_averaging = average_time_series(time_series, window_size_10, True)
select_time_series_10_averaging = time_series_10_averaging[select_vars, :]
print("shape of selected time series = ", select_time_series_10_averaging.shape)

# experiment 3 : averaging over 1000 days 
window_size_100 = 100
select_time_series_100_averaging = average_time_series(time_series, window_size_100, True)[select_vars,:]

# compute liang's function for all the elements in the list of experiments
# 
# experiment 0
nvar_results = np.array(compute_liang_nvar(select_time_series_0_averaging, 1, 1000))
tau_nvar = np.empty((len(select_vars), len(select_vars)))
error_tau_nvar = np.empty((len(select_vars), len(select_vars)))
r_nvar = np.empty((len(select_vars), len(select_vars)))
error_r_nvar = np.empty((len(select_vars), len(select_vars)))

for i in range(len(select_vars)): 
    for j in range(len(select_vars)): 
        tau_nvar[i,j] = np.abs(nvar_results[1,i,j])
        error_tau_nvar[i,j] = nvar_results[4,i,j]
        r_nvar[i,j] = nvar_results[2,i,j]
        error_r_nvar[i,j] = nvar_results[5,i,j]

fig_nvar, axs_nvar = plt.subplots(1, 2, figsize=(12, 6))
#fig_nvar.suptitle(r"Results for $d = 1e^{-8}$, weak coupling") 
#fig_nvar.suptitle(r"Results for $d = 1.1e^{-7}$, strong coupling")
#fig_nvar.suptitle(r"Results for $d = 1e^{-9}$, weak coupling")
fig_nvar.suptitle(r"Results for $d = 1.1e^{-7}$, strong coupling")

# Define the bin positions and labels
num_bins = len(select_vars)
bin_positions = np.arange(num_bins)  # Indices for the middle of the bins
bin_labels = [str(i) for i in select_vars]  # Labels (1-based indices)

# Tau nvar plot
cax1_nvar = axs_nvar[0].imshow(tau_nvar, aspect='auto', cmap='Greens')
axs_nvar[0].set_title('Tau21 Matrix (4 variables)')
axs_nvar[0].set_xlabel('Variable Index')
axs_nvar[0].set_ylabel('Variable Index')
axs_nvar[0].set_xticks(bin_positions)
axs_nvar[0].set_xticklabels(bin_labels)
axs_nvar[0].set_yticks(bin_positions)
axs_nvar[0].set_yticklabels(bin_labels)
fig_nvar.colorbar(cax1_nvar, ax=axs_nvar[0])
#add_annotations(axs_nvar[0], tau_nvar)

# R nvar plot
vmax = max(abs(r_nvar.min()), abs(r_nvar.max()))  # Symmetric color limits around zero
cax2_nvar = axs_nvar[1].imshow(r_nvar, aspect='auto', cmap='RdBu', vmin=-vmax, vmax=vmax)
axs_nvar[1].set_title('R Matrix (4 variables)')
axs_nvar[1].set_xlabel('Variable Index')
axs_nvar[1].set_ylabel('Variable Index')
axs_nvar[1].set_xticks(bin_positions)
axs_nvar[1].set_xticklabels(bin_labels)
axs_nvar[1].set_yticks(bin_positions)
axs_nvar[1].set_yticklabels(bin_labels)
fig_nvar.colorbar(cax2_nvar, ax=axs_nvar[1])
#add_annotations(axs_nvar[1], r_nvar)

# experiment 1
nvar_results1 = np.array(compute_liang_nvar(select_time_series_10_averaging, 1, 1000))

tau_nvar11 = np.empty((len(select_vars), len(select_vars)))
error_tau_nvar11 = np.empty((len(select_vars), len(select_vars)))
r_nvar11 = np.empty((len(select_vars), len(select_vars)))
error_r_nvar11 = np.empty((len(select_vars), len(select_vars)))

for i1 in range(len(select_vars)): 
    for j1 in range(len(select_vars)): 
        tau_nvar11[i1, j1] = np.abs(nvar_results1[1, i1, j1])
        error_tau_nvar11[i1, j1] = nvar_results1[4, i1, j1]
        r_nvar11[i1, j1] = nvar_results1[2, i1, j1]
        error_r_nvar11[i1, j1] = nvar_results1[5, i1, j1]

fig_nvar1, axs_nvar1 = plt.subplots(1, 2, figsize=(12, 6))
#fig_nvar1.suptitle(r"Results for $d = 1e^{-8}$, weak coupling") 
#fig_nvar1.suptitle(r"Results for $d = 1.1e^{-7}$, strong coupling")
#fig_nvar1.suptitle(r"Results for $d = 1e^{-9}$, weak coupling")
fig_nvar1.suptitle(r"Results for $d = 1.1e^{-7}$, strong coupling")

# Define the bin positions and labels
num_bins1 = len(select_vars)
bin_positions1 = np.arange(num_bins1)  # Indices for the middle of the bins
bin_labels1 = [str(i1) for i1 in select_vars]  # Labels (1-based indices)

# Tau nvar plot
cax1_nvar1 = axs_nvar1[0].imshow(tau_nvar11, aspect='auto', cmap='Greens')
axs_nvar1[0].set_title('Tau21 Matrix (4 variables)')
axs_nvar1[0].set_xlabel('Variable Index')
axs_nvar1[0].set_ylabel('Variable Index')
axs_nvar1[0].set_xticks(bin_positions1)
axs_nvar1[0].set_xticklabels(bin_labels1)
axs_nvar1[0].set_yticks(bin_positions1)
axs_nvar1[0].set_yticklabels(bin_labels1)
fig_nvar1.colorbar(cax1_nvar1, ax=axs_nvar1[0])
#add_annotations(axs_nvar1[0], tau_nvar11)

# R nvar plot
vmax1 = max(abs(r_nvar11.min()), abs(r_nvar11.max()))  # Symmetric color limits around zero
cax2_nvar1 = axs_nvar1[1].imshow(r_nvar11, aspect='auto', cmap='RdBu', vmin=-vmax1, vmax=vmax1)
axs_nvar1[1].set_title('R Matrix (4 variables)')
axs_nvar1[1].set_xlabel('Variable Index')
axs_nvar1[1].set_ylabel('Variable Index')
axs_nvar1[1].set_xticks(bin_positions1)
axs_nvar1[1].set_xticklabels(bin_labels1)
axs_nvar1[1].set_yticks(bin_positions1)
axs_nvar1[1].set_yticklabels(bin_labels1)
fig_nvar1.colorbar(cax2_nvar1, ax=axs_nvar1[1])
#add_annotations(axs_nvar1[1], r_nvar11)

# experiment 2
nvar_results2 = np.array(compute_liang_nvar(select_time_series_100_averaging, 1, 1000))

tau_nvar12 = np.empty((len(select_vars), len(select_vars)))
error_tau_nvar12 = np.empty((len(select_vars), len(select_vars)))
r_nvar12 = np.empty((len(select_vars), len(select_vars)))
error_r_nvar12 = np.empty((len(select_vars), len(select_vars)))

for i2 in range(len(select_vars)): 
    for j2 in range(len(select_vars)): 
        tau_nvar12[i2, j2] = np.abs(nvar_results2[1, i2, j2])
        error_tau_nvar12[i2, j2] = nvar_results2[4, i2, j2]
        r_nvar12[i2, j2] = nvar_results2[2, i2, j2]
        error_r_nvar12[i2, j2] = nvar_results2[5, i2, j2]

fig_nvar2, axs_nvar2 = plt.subplots(1, 2, figsize=(12, 6))
#fig_nvar2.suptitle(r"Results for $d = 1e^{-8}$, weak coupling") 
#fig_nvar2.suptitle(r"Results for $d = 1.1e^{-7}$, strong coupling")
#fig_nvar2.suptitle(r"Results for $d = 1e^{-9}$, weak coupling")
fig_nvar2.suptitle(r"Results for $d = 1.1e^{-7}$, strong coupling")

# Define the bin positions and labels
num_bins2 = len(select_vars)
bin_positions2 = np.arange(num_bins)  # Indices for the middle of the bins
bin_labels2 = [str(i2) for i2 in select_vars]  # Labels (1-based indices)

# Tau nvar plot
cax1_nvar2 = axs_nvar2[0].imshow(tau_nvar12, aspect='auto', cmap='Greens')
axs_nvar2[0].set_title('Tau21 Matrix (4 variables)')
axs_nvar2[0].set_xlabel('Variable Index')
axs_nvar2[0].set_ylabel('Variable Index')
axs_nvar2[0].set_xticks(bin_positions2)
axs_nvar2[0].set_xticklabels(bin_labels2)
axs_nvar2[0].set_yticks(bin_positions2)
axs_nvar2[0].set_yticklabels(bin_labels2)
fig_nvar2.colorbar(cax1_nvar2, ax=axs_nvar2[0])
#add_annotations(axs_nvar2[0], tau_nvar12)

# R nvar plot
vmax2 = max(abs(r_nvar12.min()), abs(r_nvar12.max()))  # Symmetric color limits around zero
cax2_nvar2 = axs_nvar2[1].imshow(r_nvar12, aspect='auto', cmap='RdBu', vmin=-vmax2, vmax=vmax2)
axs_nvar2[1].set_title('R Matrix (4 variables)')
axs_nvar2[1].set_xlabel('Variable Index')
axs_nvar2[1].set_ylabel('Variable Index')
axs_nvar2[1].set_xticks(bin_positions2)
axs_nvar2[1].set_xticklabels(bin_labels2)
axs_nvar2[1].set_yticks(bin_positions2)
axs_nvar2[1].set_yticklabels(bin_labels2)
fig_nvar2.colorbar(cax2_nvar2, ax=axs_nvar2[1])
#add_annotations(axs_nvar2[1], r_nvar12)


plt.show()

"""
# saving data with errors to file 
output_file = "tau_r_results.csv"
# Create rows for the CSV
csv_rows = []
for i in range(len(select_vars)):
    for j in range(len(select_vars)):
        row_name = f"tau{i + 1}->{j + 1}"
        row_data = [
            row_name,
            tau_nvar[i, j],
            error_tau_nvar[i, j],
            r_nvar[i, j],
            error_r_nvar[i, j]
        ]
        csv_rows.append(row_data)

# Define column headers
csv_headers = ["Pair", "tau12", "error tau12", "r", "error r"]

# Write data to CSV
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(csv_headers)
    writer.writerows(csv_rows)

print(f"Results saved to {output_file}")

"""

end = time.time()
print("execution time = ", end - start)
