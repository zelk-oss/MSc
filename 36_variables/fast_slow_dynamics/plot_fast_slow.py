"""
Here I plot the results of LKIF and TE fast and slow separate dynamics.
Bivariate case 
FAST -> chaotic variability 
SLOW -> slower and more predictable dynamics 
--- will the causality be different?  
A masking function is implemented to remove non-significant values and extremal values 
(actually this has to be modified and made better)
The matrices are rescaled with respect to a global maximum for comparison purposes 
enjoy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, LinearSegmentedColormap

sys.path.insert(0, '/home/chiaraz/thesis')
from functions_for_maooam import apply_masking

# Function to load and preprocess the CSV data
def load_data(filename, source_subset=None, target_subset=None, tau_range=None, r_range=None):
    data = pd.read_csv(filename)  # Load the file
    data.columns = data.columns.str.strip()  # Strip whitespace from column names

    # Rename columns for standardization
    data.rename(
        columns={
            "InfoFlow": "InfoFlow",
            "Error_InfoFlow": "Error_InfoFlow",
            "Tau": "Tau", 
            "Error_Tau": "Error_Tau", 
            "R": "R", 
            "Error_R": "Error_R",
            "Source": "Source", 
            "Target": "Target"
        },
        inplace=True
    )

    # Apply filtering
    if source_subset:
        data = data[data["Source"].str.replace("Var", "").astype(int).isin(source_subset)]
    if target_subset:
        data = data[data["Target"].str.replace("Var", "").astype(int).isin(target_subset)]
    if tau_range:
        data = data[(data["Tau"] >= tau_range[0]) & (data["Tau"] <= tau_range[1])]
    if r_range:
        data = data[(data["R"] >= r_range[0]) & (data["R"] <= r_range[1])]
    
    return data

# Function to plot matrices
# change c map here if you don't want the same one
def plot_matrix(matrices, labels, title, cmap, norm, xlabel, ylabel):
    num_matrices = len(matrices)
    fig, axs = plt.subplots(1, num_matrices, figsize=(5 * num_matrices, 5))
    fig.suptitle(title)

    for i, (matrix, label) in enumerate(zip(matrices, labels)):
        cax = axs[i].imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        axs[i].set_title(label)
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        num_vars = 36
        n = 5  # Frequency of ticks
        ticks = np.arange(0, num_vars, n)
        tick_labels = np.arange(1, num_vars + 1, n)
        axs[i].set_xticks(ticks)
        axs[i].set_yticks(ticks)
        axs[i].set_xticklabels(tick_labels)
        axs[i].set_yticklabels(tick_labels)
        fig.colorbar(cax, ax=axs[i])

    plt.tight_layout()
    plt.show()

# File names
file_names = {"Fast - strong coupling": "results_fast_slow/results_fast_strong_0avg.csv", 
               "Fast - strong coupling, with removal": "results_fast_slow/REMOVEDresults_fast_strong_0avg.csv",
               "Fast - weak coupling": "results_fast_slow/results_fast_weak_0avg.csv", 
               "Fast - weak coupling, with removal":"results_fast_slow/REMOVEDresults_fast_weak_0avg.csv"
               }

# Load and process data with masking
data = {}
matrices_tau = {}
matrices_r = {}
for label, file in file_names.items():
    df = load_data(file)
    tau_matrix, r_matrix, _, _ = apply_masking(df, 0.999, use_masking=True)
    data[label] = df
    # the empty matrix becomes the result of apply masking
    matrices_tau[label] = tau_matrix    
    matrices_r[label] = r_matrix


# Rescale Tau matrices after removing extreme values
rescaled_tau_matrices = {}

# Flatten all matrices and remove extreme values before finding the global max
all_values = np.concatenate([matrix.flatten() for matrix in matrices_tau.values()])
global_max_tau = np.nanmax(np.abs(all_values))  # Global max after filtering

# Normalize each matrix using the global max
for key, matrix in matrices_tau.items():
    rescaled_tau_matrices[key] = matrix / global_max_tau  # Normalize globally

# Plot masked Tau matrices
norm = Normalize(vmin=-1, vmax = 1)
plot_matrix(
    matrices=[rescaled_tau_matrices[key] for key in file_names.keys()],
    labels=list(file_names.keys()),
    title=r"Fast dynamics: LKIF",
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["darkblue", "white", "darkorange"], N=512),
    norm=norm,
    #norm=Normalize(vmin=0, vmax=1),  # Ensure the color map is correctly scaled
    xlabel="Target Variable",
    ylabel="Source Variable",
)

# Plot masked R matrices
norm=Normalize(vmin=-1, vmax = 1)
plot_matrix(
    matrices=[matrices_r[key] for key in file_names.keys()],
    labels=list(file_names.keys()),
    title=r"Fast dynamics: correlation",
    cmap="PiYG",
    norm=norm,
    xlabel="Target Variable",
    ylabel="Source Variable",
)
