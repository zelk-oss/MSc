import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import Normalize, LinearSegmentedColormap

# Add the path to the module with apply_masking
sys.path.insert(0, '/home/chiaraz/thesis')
from functions_for_maooam import apply_masking

# ------------------------
# Data Loading Functions
# ------------------------

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

        # Rename columns if needed
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
    max_tau = max(np.max(np.abs(matrix)) for matrix in matrices_tau.values() if matrix.size > 0)
    rescaled_tau_matrices = {key: matrix / max_tau for key, matrix in matrices_tau.items()} if max_tau > 0 else matrices_tau

    # Make all values positive
    for key in rescaled_tau_matrices:
        rescaled_tau_matrices[key] = np.abs(rescaled_tau_matrices[key])
    for key in matrices_r:
        matrices_r[key] = np.abs(matrices_r[key])

    return rescaled_tau_matrices, matrices_r

def load_TE_data(filename, pvalue):
    """
    Load and process TE data from a CSV file without individual normalization.
    
    Args:
        filename (str): Path to the CSV file.
        pvalue (float): p-value threshold for masking.
    
    Returns:
        np.ma.MaskedArray: The processed TE matrix (absolute values only).
    """
    data = pd.read_csv(filename)
    data.columns = data.columns.str.strip()

    sources = data['Source'].unique()
    destinations = data['Destination'].unique()
    num_vars = max(len(sources), len(destinations))

    te_matrix = np.zeros((num_vars, num_vars))
    p_value_matrix = np.zeros((num_vars, num_vars))

    for _, row in data.iterrows():
        source = int(row['Source']) - 1
        destination = int(row['Destination']) - 1
        te_value = row['TE_Kraskov']
        p_value = row['P-value']
        te_matrix[source, destination] = te_value
        p_value_matrix[source, destination] = p_value

    masked_te_matrix = np.ma.masked_where(p_value_matrix > pvalue, te_matrix)
    # Return absolute values without individual scaling
    raw_te_matrix = np.abs(masked_te_matrix)
    return raw_te_matrix

# ------------------------
# Plotting Function
# ------------------------

def plot_matrices(matrices, labels, colorbar_labels, title, cmaps, norms, xlabel, ylabel, save_path):
    """
    Plot a list of matrices side by side.
    
    Args:
        matrices (list): List of matrices to plot.
        labels (list): List of titles for each subplot.
        colorbar_labels (list): List of labels for the colorbars.
        title (str): Global title for the figure.
        cmaps (dict): Dictionary of colormaps keyed by label.
        norms (dict): Dictionary of normalization objects keyed by label.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_path (str): Path to save the resulting figure.
    """
    num_matrices = len(matrices)
    # Create the subplots without constrained_layout
    fig, axs = plt.subplots(1, num_matrices, figsize=(9 * num_matrices, 9))
    # Increase the space between subplots (wspace) and between title and subplots (top margin)
    fig.subplots_adjust(top=0.85, wspace=0.45)
    # Place the overall title with extra space above the subplots
    fig.suptitle(title, y=0.9)

    for i, (matrix, label, cbar_label) in enumerate(zip(matrices, labels, colorbar_labels)):
        cmap = cmaps.get(label)
        norm = norms.get(label)
        cax = axs[i].imshow(matrix, cmap=cmap, norm=norm, aspect="equal")
        axs[i].set_title(label)
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)

        num_vars = matrix.shape[0]
        n = max(1, num_vars // 8)
        ticks = np.arange(0, num_vars, n)
        tick_labels = np.arange(1, num_vars + 1, n)
        axs[i].set_xticks(ticks)
        axs[i].set_yticks(ticks)
        axs[i].set_xticklabels(tick_labels)
        axs[i].set_yticklabels(tick_labels)

        cbar = fig.colorbar(cax, ax=axs[i], fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
        cbar.ax.tick_params(labelsize=24)

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()

# Set global font sizes for consistency
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 26,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "figure.titlesize": 32,
    "legend.fontsize": 22,
})

# ------------------------
# TE Comparison Plot
# ------------------------
# Define file names for TE (adjust paths as needed)
filenameTE_weak = "data_biv_TE/results_weak_0days.csv"
filenameTE_strong = "data_biv_TE/results_strong_0days.csv"
save_path_TE = "/home/chiaraz/pictures_thesis/final/methods_comparison/TE_weak_strong.png"

# Load TE data for weak and strong coupling
te_weak = load_TE_data(filenameTE_weak, pvalue=0.001)
te_strong = load_TE_data(filenameTE_strong, pvalue=0.001)

# Compute global maximum across both matrices for common normalization
global_max = max(np.max(te_weak), np.max(te_strong))
rescaled_te_weak = te_weak / global_max
rescaled_te_strong = te_strong / global_max

# Plot the strong coupling on the left and weak on the right
te_matrices_to_plot = [rescaled_te_strong, rescaled_te_weak]

plot_matrices(
    matrices=te_matrices_to_plot,
    labels=[r"$d = 1.1e^{-7}$", r"$d = 1.e^{-8}$"],
    colorbar_labels=["transfer entropy", "transfer entropy"],
    title="Transfer Entropy comparison: strong and weak coupling",
    cmaps={
        r"$d = 1.1e^{-7}$": LinearSegmentedColormap.from_list("white_to_blue", ["white", "blue"], N=256),
        r"$d = 1.e^{-8}$": LinearSegmentedColormap.from_list("white_to_blue", ["white", "blue"], N=256)
    },
    norms={
        r"$d = 1.1e^{-7}$": Normalize(vmin=0, vmax=1),
        r"$d = 1.e^{-8}$": Normalize(vmin=0, vmax=1)
    },
    xlabel="Target Variable",
    ylabel="Source Variable",
    save_path=save_path_TE
)

# ------------------------
# LKIF Comparison Plot (unchanged)
# ------------------------
file_namesLKIF_weak = {"LKIF": "../averaging/results_averaging_atmosphere/liang_res_11days_0days_weak_avg.csv"}
file_namesLKIF_strong = {"LKIF": "../averaging/results_averaging_atmosphere/liang_res_11days_0days_strong_avg.csv"}
save_path_LKIF = "/home/chiaraz/thesis/pictures_thesis/final/methods_comparison/LKIF_weak_strong.png"

rescaled_tau_weak, _ = load_LKIF_data(file_namesLKIF_weak)
rescaled_tau_strong, _ = load_LKIF_data(file_namesLKIF_strong)

lkif_matrices_to_plot = [rescaled_tau_strong['LKIF'], rescaled_tau_weak['LKIF']]

plot_matrices(
    matrices=lkif_matrices_to_plot,
    labels=[r"$d = 1.1e^{-7}$", r"$d = 1.e^{-8}$"],
    colorbar_labels=["LKIF", "LKIF"],
    title="Liang-Kleeman Information Flow comparison: strong and weak coupling",
    cmaps={
        r"$d = 1.1e^{-7}$": LinearSegmentedColormap.from_list("white_to_orangered", ["white", "orangered"], N=256),
        r"$d = 1.e^{-8}$": LinearSegmentedColormap.from_list("white_to_orangered", ["white", "orangered"], N=256)
    },
    norms={
        r"$d = 1.1e^{-7}$": Normalize(vmin=0, vmax=1),
        r"$d = 1.e^{-8}$": Normalize(vmin=0, vmax=1)
    },
    xlabel="Target Variable",
    ylabel="Source Variable",
    save_path=save_path_LKIF
)
