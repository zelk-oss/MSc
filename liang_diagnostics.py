import numpy as np 
import os 
import csv 
import matplotlib.pyplot as plt  # Added for visualization

# importing Liang's multivariate formula from another folder 
import sys 
sys.path.insert(0, '/home/chiaraz/Liang_Index_climdyn')
from function_liang_nvar import compute_liang_nvar

# import functions for processing maooam data 
from functions_for_maooam import convert_days_to_points
from functions_for_maooam import average_time_series_3D
from functions_for_maooam import introduce_lag_diagnostics

"""end of functions """

# Define the folder containing the .npy files
folder_path = "../data_toobig/generated_diagnostics_3_4/"  # Replace with your folder path

# List of file names to process (base names without the "_data.npy" suffix)
file_base_names = [
    "delta_T_a", # 0
    "delta_T_o", # 1
    #"LowerLayerAtmosphericUWindDiagnostic", # 2
    #"LowerLayerAtmosphericVWindDiagnostic", # 3
    #"LowerLayerAtmosphericWindIntensityDiagnostic", # 4
    #"MiddleAtmosphericUWindDiagnostic", # 5
    #"MiddleAtmosphericVWindDiagnostic", # 6
    #"MiddleAtmosphericWindIntensityDiagnostic", # 7
    #"MiddleLayerVerticalVelocity",  # 8
    "psi_1_a", # 9
    "psi_3_a", # 10 
    "psi_a", # 11
    "psi_o", # 12
    "T_a", # 13
    "T_o", # 14 
    #"UpperLayerAtmosphericUWindDiagnostic", # 15
    #"UpperLayerAtmosphericVWindDiagnostic", # 16
    #"UpperLayerAtmosphericWindIntensityDiagnostic" # 17
]

# downsampling for faster integration 
accelerate = 1
# considered grid point for point-wise analysis 
x = 2
y = 1

"""
Data extraction and processing: 
1) file name 
2) load the .npy file. Shape = (time, x, y)
3) downsample the data and select a single point in space. Shape = (time/accelerate, )
4) add value to the dictionary 
"""
diagnostic_data = {} # dictionary. Here I will put extracted data 

# Loop through all file names and process each file
for base_name in file_base_names:
    try:
        # Construct the full file path
        file_name = os.path.join(folder_path, f"{base_name}_data.npy")
        
        # Load the data
        file_data = np.load(file_name, allow_pickle=True)

        # toggle this if you want averaging or not 
        averaging_months_window = 12
        averaged_file_data = average_time_series_3D(file_data, averaging_months_window, True)
        print("non averaged vs averaged data dimension = ", file_data.shape, averaged_file_data.shape)
        # Sample and reshape the data
        sampled_data = averaged_file_data[::accelerate, x, y] 
        diagnostic_data[base_name] = sampled_data

        # Print the result
        print(f"Processed: {base_name}, Initial shape: {file_data.shape}, Reshaped data shape: {sampled_data.shape}")
    except Exception as e:
        print(f"Error processing {base_name}: {e}")


"""
finished extracting stuff from files. Now we can use the functions as we prefer 
"""

# Combine time series into an array
select_time_series = np.array([
    # here select some functions 
    diagnostic_data["delta_T_a"], # delta T a 
    diagnostic_data["delta_T_o"], # delta T o 
    diagnostic_data["psi_1_a"], # Psi^3 a 
    diagnostic_data["psi_3_a"], # Psi^1 a 
    diagnostic_data["psi_o"] # Psi a
]) # an nvar * time array 

print("shape of time series matrix. It should be nvar * N = ", select_time_series.shape)

"""
# create matrices with results 
tau_nvar = np.empty((select_time_series.shape[0], select_time_series.shape[0]))
error_tau_nvar = np.empty((select_time_series.shape[0], select_time_series.shape[0]))
r_nvar = np.empty((select_time_series.shape[0], select_time_series.shape[0]))
error_r_nvar = np.empty((select_time_series.shape[0], select_time_series.shape[0]))

# Compute Liang indices
nvar_results = np.array(compute_liang_nvar(select_time_series, 1, 1000*averaging_months_window))

for i in range(select_time_series.shape[0]): 
    for j in range(select_time_series.shape[0]): 
        tau_nvar[i,j] = np.abs(nvar_results[1,i,j])
        error_tau_nvar[i,j] = nvar_results[4,i,j]
        r_nvar[i,j] = nvar_results[2,i,j]
        error_r_nvar[i,j] = nvar_results[5,i,j]
        print("element (", i ,",", j ,") added")

tau_nvar = np.abs(nvar_results[1])  # Directly extract the tau matrix
r_nvar = nvar_results[2] 


##############################################
############# Write data to CSV ##############
##############################################
output_file = "tau_r_results_diagnostics.csv"
# Create rows for the CSV
csv_rows = []
for i in range(select_time_series.shape[0]):
    for j in range(select_time_series.shape[0]):
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
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(csv_headers)
    writer.writerows(csv_rows)

print(f"Results saved to {output_file}")
"""

##############################################
############# Visualization ##################
##############################################
def visualize_diagnostic_time_series(diagnostic_data, diagnostics_to_plot):
    """
    Visualize the time series for selected diagnostics.

    Parameters:
    - diagnostic_data: dict containing diagnostic time series.
    - diagnostics_to_plot: list of diagnostic names to visualize.
    """
    plt.figure(figsize=(12, 6))
    for diag in diagnostics_to_plot:
        if diag in diagnostic_data:
            plt.plot(diagnostic_data[diag], label=diag)
        else:
            print(f"Warning: {diag} not found in diagnostic data.")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Diagnostic Time Series")
    plt.legend()
    plt.grid()
    plt.show()

# Example usage: Visualize specific diagnostics
diagnostics_to_plot = ["delta_T_a"]
visualize_diagnostic_time_series(diagnostic_data, diagnostics_to_plot)

diagnostics_to_plot = ["delta_T_o"]
visualize_diagnostic_time_series(diagnostic_data, diagnostics_to_plot)

diagnostics_to_plot = ["psi_a"]
visualize_diagnostic_time_series(diagnostic_data, diagnostics_to_plot)

diagnostics_to_plot = ["psi_1_a"]
visualize_diagnostic_time_series(diagnostic_data, diagnostics_to_plot)

diagnostics_to_plot = ["psi_3_a"]
visualize_diagnostic_time_series(diagnostic_data, diagnostics_to_plot)

diagnostics_to_plot = ["psi_o"]
visualize_diagnostic_time_series(diagnostic_data, diagnostics_to_plot)