import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

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

# Function to plot matrices with masking
def plot_matrix(matrices, labels, title, cmap, xlabel, ylabel):
    num_matrices = len(matrices)
    fig, axs = plt.subplots(1, num_matrices, figsize=(5 * num_matrices, 5))
    fig.suptitle(title)

    for i, (matrix, label) in enumerate(zip(matrices, labels)):
        cax = axs[i].imshow(matrix, cmap=cmap, aspect="auto")
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
file_names = {
    "original data": "../averaging/liang_res_11days_0_weak_avg.csv",
    "6 months": "liang_lagging_results/results_data11days_lag_182.625days.csv",
    "1 year": "liang_lagging_results/results_data11days_lag_365.25days.csv",
    "10 years": "liang_lagging_results/results_data11days_lag_3652.5days.csv",
}

# Load and process data with masking
data = {}
matrices_tau = {}
matrices_r = {}
for label, file in file_names.items():
    df = load_data(file)
    tau_matrix, r_matrix, _, _ = apply_masking(df, use_masking=True)
    data[label] = df
    matrices_tau[label] = tau_matrix
    matrices_r[label] = r_matrix

# Plot masked Tau matrices
plot_matrix(
    matrices=[matrices_tau[key] for key in file_names.keys()],
    labels=list(file_names.keys()),
    title=r"Atmospheric and oceanic lagged data, $d = 1.e^{-8}$: $\tau_{j \to i}$ comparison",
    cmap="Greens",
    xlabel="Target Variable",
    ylabel="Source Variable",
)

# Plot masked R matrices
plot_matrix(
    matrices=[matrices_r[key] for key in file_names.keys()],
    labels=list(file_names.keys()),
    title=r"Atmospheric and oceanic lagged data, $d = 1.e^{-8}$: R comparison",
    cmap="RdBu",
    xlabel="Target Variable",
    ylabel="Source Variable",
)
