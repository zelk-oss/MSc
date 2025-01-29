"""
Here I plot the results of LKIF multivariate computation. 
Interesting to note that the masking of non-significant values removes all the anticorrelations 
and in general it's quite impactful 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib.colors as mcolors

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

    from matplotlib.colors import Normalize

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
file_names = {"original data": "liang_res_11days_0_avg.csv", 
               "33 days": "liang_res_11days_33days_avg.csv", 
               "1 year":"liang_res_11days_1yr_avg.csv", 
               "10 years" : "liang_res_11days_10yr_avg.csv"}
avg_days = [0, 33, 365.25, 3652.5]  # Corresponding averaging periods

# Load and process data with masking
data = {}
matrices_tau = {}
matrices_r = {}
for label, file in file_names.items():
    df = load_data(file)
    tau_matrix, r_matrix, _, _ = apply_masking(df, use_masking=False)
    data[label] = df
    matrices_tau[label] = tau_matrix
    matrices_r[label] = r_matrix


# Plot masked Tau matrices
plot_matrix(
    matrices=[matrices_tau[key] for key in file_names.keys()],
    labels=list(file_names.keys()),
    title=r"Atmospheric and oceanic averaged data, $d = 1.1e^{-7}$: $\tau_{j \to i}$ comparison",
    cmap="Greens",
    xlabel="Target Variable",
    ylabel="Source Variable",
)

# Plot masked R matrices
plot_matrix(
    matrices=[matrices_r[key] for key in file_names.keys()],
    labels=list(file_names.keys()),
    title=r"Atmospheric and oceanic averaged data, $d = 1.1e^{-7}$: R comparison",
    cmap="RdBu",
    xlabel="Target Variable",
    ylabel="Source Variable",
)

"""

# Save differences to file and plot differences
output_file = "data11days_weak_averaged_DIFFERENCES.txt"
with open(output_file, "w") as f:
    f.write("Pairwise differences for Tau and R values:\n")

    for i in range(len(avg_days)):
        for j in range(i + 1, len(avg_days)):
            tau_diff = np.abs((tau_matrices[i] - tau_matrices[j]) / ((tau_matrices[i] + tau_matrices[j])/2))
            r_diff = np.abs((r_matrices[i] - r_matrices[j]) / ((r_matrices[i] + r_matrices[j])/2))

            # Write to file
            f.write(f"Relative differences between {avg_days[i]} and {avg_days[j]} days averaging:\n")
            f.write("Tau relative differences:\n")
            np.savetxt(f, tau_diff, fmt="%.5f")
            f.write("\n")
            f.write("R relative differences:\n")
            np.savetxt(f, r_diff, fmt="%.5f")
            f.write("\n\n")

            # Plot differences
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f"Relative differences between {avg_days[i]} and {avg_days[j]} days averaging")

            # Tau differences plot
            cax1 = axs[0].imshow(tau_diff, aspect="auto", cmap=plt.cm.Greens)
            axs[0].set_title("Tau relative differences")
            axs[0].set_xlabel("Target Variable")
            axs[0].set_ylabel("Source Variable")
            num_vars = tau_diff.shape[0]
            n = 5  # Frequency of ticks
            ticks = np.arange(0, num_vars, n)
            tick_labels = np.arange(1, num_vars + 1, n)
            axs[0].set_xticks(ticks)
            axs[0].set_yticks(ticks)
            axs[0].set_xticklabels(tick_labels)
            axs[0].set_yticklabels(tick_labels)
            fig.colorbar(cax1, ax=axs[0])

            # R differences plot
            cax2 = axs[1].imshow(r_diff, aspect="auto", cmap=plt.cm.Blues)
            axs[1].set_title("R relative differences")
            axs[1].set_xlabel("Target Variable")
            axs[1].set_ylabel("Source Variable")
            axs[1].set_xticks(ticks)
            axs[1].set_yticks(ticks)
            axs[1].set_xticklabels(tick_labels)
            axs[1].set_yticklabels(tick_labels)
            fig.colorbar(cax2, ax=axs[1])

            plt.tight_layout()
            plt.show()

# Plot Tau vs R matrices side-by-side for each case
for idx, days in enumerate(avg_days):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Comparison of Tau and R for {days} Days Averaging")

    # Tau plot
    tau = tau_matrices[idx]
    cmap_tau = plt.cm.Greens
    cmap_tau.set_bad(color="white")
    cax1 = axs[0].imshow(tau, aspect="auto", cmap=cmap_tau)
    axs[0].set_title(rf"$\tau_{{12}}$ for {days} days")
    axs[0].set_xlabel("Target Variable")
    axs[0].set_ylabel("Source Variable")
    num_vars = tau.shape[0]
    n = 5  # Frequency of ticks
    ticks = np.arange(0, num_vars, n)
    tick_labels = np.arange(1, num_vars + 1, n)
    axs[0].set_xticks(ticks)
    axs[0].set_yticks(ticks)
    axs[0].set_xticklabels(tick_labels)
    axs[0].set_yticklabels(tick_labels)
    fig.colorbar(cax1, ax=axs[0])

    # R plot
    r = r_matrices[idx]
    cmap_r = plt.cm.RdBu
    cmap_r.set_bad(color="white")
    vmax = max(abs(r.min()), abs(r.max()))
    cax2 = axs[1].imshow(r, aspect="auto", cmap=cmap_r, vmin=-vmax, vmax=vmax)
    axs[1].set_title(f"R for {days} days")
    axs[1].set_xlabel("Target Variable")
    axs[1].set_ylabel("Source Variable")
    axs[1].set_xticks(ticks)
    axs[1].set_yticks(ticks)
    axs[1].set_xticklabels(tick_labels)
    axs[1].set_yticklabels(tick_labels)
    fig.colorbar(cax2, ax=axs[1])

    plt.tight_layout()

# Plot all Tau matrices side-by-side
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(r"Comparison of $\tau_{12}$ values across averaging periods")

for idx, days in enumerate(avg_days):
    tau = tau_matrices[idx]
    cmap_tau = plt.cm.Greens
    cax = axs[idx].imshow(tau, aspect="auto", cmap=cmap_tau)
    axs[idx].set_title(rf"$\tau_{{12}}$ for {days} days")
    axs[idx].set_xlabel("Target Variable")
    axs[idx].set_ylabel("Source Variable")
    axs[idx].set_xticks(ticks)
    axs[idx].set_yticks(ticks)
    axs[idx].set_xticklabels(tick_labels)
    axs[idx].set_yticklabels(tick_labels)
    fig.colorbar(cax, ax=axs[idx])

plt.tight_layout()

# Plot all R matrices side-by-side
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Comparison of R Matrices Across Averaging Periods")

for idx, days in enumerate(avg_days):
    r = r_matrices[idx]
    cmap_r = plt.cm.RdBu
    vmax = max(abs(r.min()), abs(r.max()))
    cax = axs[idx].imshow(r, aspect="auto", cmap=cmap_r, vmin=-vmax, vmax=vmax)
    axs[idx].set_title(f"R for {days} days")
    axs[idx].set_xlabel("Target Variable")
    axs[idx].set_ylabel("Source Variable")
    axs[idx].set_xticks(ticks)
    axs[idx].set_yticks(ticks)
    axs[idx].set_xticklabels(tick_labels)
    axs[idx].set_yticklabels(tick_labels)
    fig.colorbar(cax, ax=axs[idx])

plt.tight_layout()
plt.show()
    """

