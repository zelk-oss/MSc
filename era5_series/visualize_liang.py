import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import Normalize

sys.path.insert(0, '/home/chiaraz/thesis')
from functions_for_maooam import apply_masking

# Function to load and preprocess the CSV data
def load_data(filename, source_subset=None, target_subset=None, tau_range=None, r_range=None):
    data = pd.read_csv(filename)  # Load the file
    data.columns = data.columns.str.strip()  # Strip whitespace from column names

    # Standardize column names
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

    # Apply optional filters
    if source_subset:
        data = data[data["Source"].str.replace("Var", "").astype(int).isin(source_subset)]
    if target_subset:
        data = data[data["Target"].str.replace("Var", "").astype(int).isin(target_subset)]
    if tau_range:
        data = data[(data["Tau"] >= tau_range[0]) & (data["Tau"] <= tau_range[1])]
    if r_range:
        data = data[(data["R"] >= r_range[0]) & (data["R"] <= r_range[1])]
    
    return data


# Function to plot Liang results in matrix form
def plot_matrix_with_values(matrix, pvals=None, regions=None, title="", cmap="Reds",
                            vmin=None, vmax=None, sig_level=0.05,
                            xlabel="Target Variable", ylabel="Source Variable"):
    """
    Plot a matrix as a heatmap with numeric values inside each cell.
    Marks cells with '*' if not significant (based on pvals).
    
    Args:
        matrix (ndarray): values to plot
        pvals (ndarray): p-values (same shape as matrix), optional
        regions (list): labels for axes
        title (str): plot title
        cmap (str): colormap
        vmin, vmax (float): color normalization
        sig_level (float): significance threshold
    """
    nvar = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(matrix, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))

    # Add text annotations
    for i in range(nvar):
        for j in range(nvar):
            val = matrix[i, j]
            text = f"{val:.3f}"
            if pvals is not None and i != j:
                if pvals[i, j] > sig_level:  # mark non-significant
                    text += " *"
            ax.text(j, i, text, ha="center", va="center", color="black")

    # Configure axes
    ax.set_xticks(range(nvar))
    ax.set_yticks(range(nvar))
    if regions is not None:
        ax.set_xticklabels(regions, rotation=45, ha="left", rotation_mode="anchor")
        ax.set_yticklabels(regions)
    else:
        ax.set_xticklabels(range(1, nvar + 1))
        ax.set_yticklabels(range(1, nvar + 1))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.colorbar(cax, ax=ax, label="Value")
    plt.tight_layout()
    plt.show()


# === Main ===
filename = "liang_results.csv"

# Load and mask
df = load_data(filename)
tau_matrix, r_matrix, _, _ = apply_masking(df, use_masking=True)
for i in range(4): 
    tau_matrix[i,i] = 0
    r_matrix[i,i] = 0

# Rescale Tau by global maximum
global_max_tau = np.nanmax(np.abs(tau_matrix))
rescaled_tau_matrix = tau_matrix / global_max_tau

# Define region names explicitly
regions = ["EPAC", "CPAC", "NAT", "WPAC"]

# Plot Tau
plot_matrix_with_values(
    matrix=rescaled_tau_matrix,
    pvals=None,  
    regions=regions,
    title=r"$\tau_{j \to i}$ (normalized)",
    cmap="coolwarm",
    vmin=-1, vmax=1
)

# Plot R
plot_matrix_with_values(
    matrix=r_matrix,
    pvals=None,
    regions=regions,
    title=r"Correlation",
    cmap="coolwarm",
    vmin=-1, vmax=1
)
