"""
This script reads the r and tau data for the diagnostic functions from a .csv file 
and plots the results producing coloured fun matrices. 
* check the masking method, I need to implement the statistically relevance toggle better 
with hypothesis testing --> check python stats
* add notations when appropriate 
"""
import numpy as np
import matplotlib.pyplot as plt
import csv

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

# Read data from CSV
input_file = "tau_r_results_diagnostics.csv"
data = []
with open(input_file, mode="r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        data.append(row)

# Extract matrix dimensions
num_vars = int(np.sqrt(len(data)))
tau_nvar = np.zeros((num_vars, num_vars))
error_tau_nvar = np.zeros((num_vars, num_vars))
r_nvar = np.zeros((num_vars, num_vars))
error_r_nvar = np.zeros((num_vars, num_vars))

# Populate matrices from CSV data
for row in data:
    pair, tau, error_tau, r, error_r = row
    i, j = map(int, pair[3:].split("->"))
    tau_nvar[i - 1, j - 1] = float(tau)
    error_tau_nvar[i - 1, j - 1] = float(error_tau)
    r_nvar[i - 1, j - 1] = float(r)
    error_r_nvar[i - 1, j - 1] = float(error_r)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Results Visualization")

bin_positions = np.arange(num_vars)
bin_labels = [r"$\delta T_a$", r"$\delta T_o$", r"$\Psi^1_a$", r"$\Psi^3_a$", r"$\Psi_o$"]

# Plot tau matrix
tau_mask = error_tau_nvar > tau_nvar
masked_tau = np.copy(tau_nvar)
masked_tau[tau_mask] = np.nan
cax1 = axs[0].imshow(masked_tau, aspect='auto', cmap='Greens', interpolation='none')
axs[0].set_title(r'$\tau_{2 \to 1}$ Matrix')
axs[0].set_xticks(bin_positions)
axs[0].set_xticklabels(bin_labels)
axs[0].set_yticks(bin_positions)
axs[0].set_yticklabels(bin_labels)
fig.colorbar(cax1, ax=axs[0])
add_annotations(axs[0], tau_nvar, tau_mask)  # Add annotations with the mask

# Plot r matrix
r_mask = error_r_nvar > np.abs(r_nvar)
masked_r = np.copy(r_nvar)
masked_r[r_mask] = np.nan
vmax = max(abs(r_nvar.min()), abs(r_nvar.max()))
cax2 = axs[1].imshow(masked_r, aspect='auto', cmap='RdBu', vmin=-vmax, vmax=vmax, interpolation='none')
axs[1].set_title(r'R Matrix')
axs[1].set_xticks(bin_positions)
axs[1].set_xticklabels(bin_labels)
axs[1].set_yticks(bin_positions)
axs[1].set_yticklabels(bin_labels)
fig.colorbar(cax2, ax=axs[1])
add_annotations(axs[1], r_nvar, r_mask)  # Add annotations with the mask

plt.show()
