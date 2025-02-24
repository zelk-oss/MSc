"""
In this script we plot side by side TE, LKIF and R. 
First step: plotting the non averaged data, everything normalized to 0,1.
There will also be some analysis of the errors etc. 
I keep the style of TE but with the cool features of LKIF that I worked on. 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, LinearSegmentedColormap

sys.path.insert(0, '/home/chiaraz/thesis')
from functions_for_maooam import apply_masking

# LKIF: Function to load and process the CSV data into tau and r masked matrices
def load_LKIF_data(file_names, source_subset=None, target_subset=None, tau_range=None, r_range=None):
    """
    Load, filter, and process LKIF data from given files.
    
    Args:
        file_names (dict): Dictionary of filenames with labels as keys.
        source_subset (list, optional): Subset of source variables to include.
        target_subset (list, optional): Subset of target variables to include.
        tau_range (tuple, optional): Min and max range for Tau filtering.
        r_range (tuple, optional): Min and max range for R filtering.
    
    Returns:
        dict: Rescaled tau matrices.
        dict: r matrices.
    """
    data = {}
    matrices_tau = {}
    matrices_r = {}

    for label, filename in file_names.items():
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()  # Remove whitespace from column names

        # Rename columns
        df.rename(
            columns={
                "InfoFlow": "InfoFlow",
                "Error_InfoFlow": "Error_InfoFlow",
                "Tau": "Tau",
                "Error_Tau": "Error_Tau",
                "R": "R",
                "Error_R": "Error_R",
                "Source": "Source",
                "Target": "Target",
            },
            inplace=True,
        )

        # Apply filtering
        if source_subset:
            df = df[df["Source"].str.replace("Var", "").astype(int).isin(source_subset)]
        if target_subset:
            df = df[df["Target"].str.replace("Var", "").astype(int).isin(target_subset)]
        if tau_range:
            df = df[(df["Tau"] >= tau_range[0]) & (df["Tau"] <= tau_range[1])]
        if r_range:
            df = df[(df["R"] >= r_range[0]) & (df["R"] <= r_range[1])]

        # Apply masking
        tau_matrix, r_matrix, _, _ = apply_masking(df, threshold_for_extremes=0.999, use_masking=True)
        data[label] = df
        matrices_tau[label] = tau_matrix
        matrices_r[label] = r_matrix

    # Rescale Tau matrices
    max_tau = max(np.max(np.abs(matrix)) for matrix in matrices_tau.values() if matrix.size > 0)  # Avoid empty matrices
    rescaled_tau_matrices = {key: matrix / max_tau for key, matrix in matrices_tau.items()} if max_tau > 0 else matrices_tau

    # make all the values positive 
    for key in rescaled_tau_matrices:
        rescaled_tau_matrices[key] = np.abs(rescaled_tau_matrices[key])

    for key in matrices_r:
        matrices_r[key] = np.abs(matrices_r[key])

    min_tau = min(np.min(matrix) for matrix in rescaled_tau_matrices.values())
    print("min of Tau", min_tau)

    return rescaled_tau_matrices, matrices_r

# Extract data from TE file and rescale 
def load_TE_data(filename, pvalue):
    # Read the input file
    data = pd.read_csv(filename)

    # Normalize column names
    data.columns = data.columns.str.strip()

    # Get source and destination range
    sources = data['Source'].unique()
    destinations = data['Destination'].unique()
    num_vars = max(len(sources), len(destinations))  # Fix for correct indexing

    # Initialize matrices
    te_matrix = np.zeros((num_vars, num_vars))
    p_value_matrix = np.zeros((num_vars, num_vars))

    # Populate the matrices
    for _, row in data.iterrows():
        source = int(row['Source']) - 1  # Convert 1-based to 0-based
        destination = int(row['Destination']) - 1
        te_value = row['TE_Kraskov']
        p_value = row['P-value']
        te_matrix[source, destination] = te_value
        p_value_matrix[source, destination] = p_value

    # Apply masking
    masked_te_matrix = np.ma.masked_where(p_value_matrix > pvalue, te_matrix)
    # rescale value to -1, 1
    max_te = np.max(np.abs(masked_te_matrix))
    rescaled_masked_te_matrix = masked_te_matrix / np.abs(max_te)

    # take absolute value of the matrix 
    rescaled_masked_te_matrix = np.abs(rescaled_masked_te_matrix)
    print("rescaled TE", rescaled_masked_te_matrix)

    return rescaled_masked_te_matrix  # Returns a 36×36 masked array


###########!!!!!!!!!!!!#############
## act here to change fonts etc 

# Set global font sizes for better readability
plt.rcParams.update({
    "font.size": 20,           # General font size
    "axes.titlesize": 28,      # Title above each subplot
    "axes.labelsize": 22,      # X and Y labels (Target Variable & Source Variable)
    "xtick.labelsize": 20,     # X-axis tick labels
    "ytick.labelsize": 20,     # Y-axis tick labels
    "figure.titlesize": 32,    # Global figure title
    "legend.fontsize": 22,     # Legend font size (if used)
})


# Function to plot matrices side by side
def plot_matrices(matrices, labels, colorbar_labels, title, cmaps, norms, xlabel, ylabel, save_path):
    num_matrices = len(matrices)
    fig, axs = plt.subplots(1, num_matrices, figsize=(9 * num_matrices, 9), constrained_layout=True)
    fig.suptitle(title)

    for i, (matrix, label, cbar_label) in enumerate(zip(matrices, labels, colorbar_labels)):
        cmap = cmaps.get(label)  # Choose colormap based on matrix type
        norm = norms.get(label)
        cax = axs[i].imshow(matrix, cmap=cmap, norm=norm, aspect="equal")  # Ensuring square aspect ratio

        # Set title above each matrix
        axs[i].set_title(label)
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)

        # Set tick marks
        num_vars = matrix.shape[0]
        n = max(1, num_vars // 8)  # Dynamically choose tick frequency
        ticks = np.arange(0, num_vars, n)
        tick_labels = np.arange(1, num_vars + 1, n)
        axs[i].set_xticks(ticks)
        axs[i].set_yticks(ticks)
        axs[i].set_xticklabels(tick_labels)
        axs[i].set_yticklabels(tick_labels)

        # Assign different labels for colorbars
        cbar = fig.colorbar(cax, ax=axs[i], fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
        cbar.ax.tick_params(labelsize=24)

    # Save the figure
    plt.savefig(save_path, dpi=300)
    plt.show()

# File names
file_namesLKIF = {"LKIF": "../averaging/results_averaging_atmosphere/liang_res_11days_0days_weak_avg.csv"}           
filenameTE = "data_biv_TE/results_weak_0days.csv"

# File save path
save_path = "/home/chiaraz/pictures_thesis/final/methods_comparison/weak_0days_avg.png"


# Extract tau and r matrices
rescaled_tau, r = load_LKIF_data(file_namesLKIF)
# Extract TE matrix
rescaled_te = load_TE_data(filenameTE, pvalue=0.01)

# Organizing matrices to plot
matrices_to_plot = [rescaled_tau['LKIF'], rescaled_te, r['LKIF']]

# Define colormap per matrix type
colormap_dict = {
    "Liang-Kleeman IF": LinearSegmentedColormap.from_list(
    "white_to_color", ["white", "orangered"], N=256
    ),
    "Transfer Entropy": LinearSegmentedColormap.from_list(
    "white_to_color", ["white", "blue"], N=256
    ),
    "Correlation": LinearSegmentedColormap.from_list(
    "white_to_color", ["white", "black"], N=256
    )
}

colorbar_labels = ["LKIF", "TE", "Correlation"]

# Define normalization for a specific matrix (e.g., TE)
norm_dict = {
    "LKIF": Normalize(vmin=0, vmax=1),
    "TE": Normalize(vmin=0, vmax=1),
    "Correlation": Normalize(vmin= 0, vmax = 1)
}

# Plot matrices with the updated function
plot_matrices(
    matrices=matrices_to_plot,
    labels=["Liang-Kleeman IF", "Transfer Entropy", "Correlation"],  
    colorbar_labels=colorbar_labels,  
    title=r"No running mean; $d = 1.e^{-8}$",
    cmaps=colormap_dict,
    norms=norm_dict,
    xlabel="Target Variable",
    ylabel="Source Variable",
    save_path=save_path
)

# TODO: do the same for average once you understand which to plot 
