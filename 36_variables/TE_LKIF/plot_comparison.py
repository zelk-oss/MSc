"""
In this script we plot side by side TE, LKIF and R. 
First step: plotting the non averaged data, everything normalized to -1,1.
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
        print("Te value: ", te_value)
        p_value_matrix[source, destination] = p_value

    # Apply masking
    masked_te_matrix = np.ma.masked_where(p_value_matrix > pvalue, te_matrix)
    print("masked matrix", masked_te_matrix)
    # rescale value to -1, 1
    max_te = np.max(np.abs(masked_te_matrix))
    rescaled_masked_te_matrix = masked_te_matrix / max_te
    print("rescaled te matrix", rescaled_masked_te_matrix)
    return rescaled_masked_te_matrix  # Returns a 36×36 masked array

# Function to plot matrices side by side
def plot_matrices(matrices, labels, title, cmaps, norms, xlabel, ylabel):
    num_matrices = len(matrices)
    fig, axs = plt.subplots(1, num_matrices, figsize=(5 * num_matrices, 5))
    fig.suptitle(title)

    for i, (matrix, label) in enumerate(zip(matrices, labels)):
        cmap = cmaps.get(label)  # Choose colormap based on matrix type
        norm = norms.get(label)
        cax = axs[i].imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        axs[i].set_title(label)
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)

        # Set tick marks
        num_vars = 36
        n = 5  # Frequency of ticks
        ticks = np.arange(0, num_vars, n)
        tick_labels = np.arange(1, num_vars + 1, n)
        axs[i].set_xticks(ticks)
        axs[i].set_yticks(ticks)
        axs[i].set_xticklabels(tick_labels)
        axs[i].set_yticklabels(tick_labels)

        fig.colorbar(cax, ax=axs[i], label=label)

    plt.tight_layout()
    plt.show()

# File names
file_namesLKIF = {"LKIF": "../averaging/results_averaging_atmosphere/liang_res_11days_100yr_strong_avg.csv"}           
filenameTE = "data_TE/results_100yr_strong_largewindow.csv"

# Extract tau and r matrices
rescaled_tau, r = load_LKIF_data(file_namesLKIF)
# Extract TE matrix
rescaled_te = load_TE_data(filenameTE, pvalue=0.01)

# Organizing matrices to plot
matrices_to_plot = [rescaled_tau['LKIF'], rescaled_te, r['LKIF']]

# Define colormap per matrix type
colormap_dict = {
    r"normalized $\tau$": "PiYG",
    "normalized TE": "PiYG",
    "R": "PiYG"
}

# Define normalization for a specific matrix (e.g., TE)
norm_dict = {
    r"normalized $\tau$": Normalize(vmin=-1, vmax=1),
    "normalized TE": Normalize(vmin=-1, vmax=1),  # Only normalize TE
    "R": Normalize(vmin= -1, vmax = 1)
}

# Plot matrices
plot_matrices(
    matrices=matrices_to_plot,
    labels=[r"normalized $\tau$", "normalized TE", "R"],
    title=r"Bivariate analysis of Fourier components - 100 years running mean over atmospheric components; $d = 1.1e^{-7}$",
    cmaps=colormap_dict,
    norms=norm_dict,
    xlabel="Target Variable",
    ylabel="Source Variable",
)
