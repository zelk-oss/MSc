import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import Normalize

sys.path.insert(0, '/home/chiaraz/thesis')
from functions_for_maooam import apply_masking

def load_data(file_names, method, pvalue=None):
    matrices_list = []
    for file_name in file_names:
        df = pd.read_csv(file_name)
        df.columns = df.columns.str.strip()
        
        if method == "LKIF":
            df.rename(columns={
                "Tau": "Tau", "R": "R", "Source": "Source", "Target": "Target"
            }, inplace=True)
            tau_matrix, r_matrix, _, _ = apply_masking(df, threshold_for_extremes=0.999, use_masking=True)
            max_tau = np.max(np.abs(tau_matrix)) if tau_matrix.size > 0 else 1
            rescaled_tau_matrix = np.abs(tau_matrix / max_tau) if max_tau > 0 else tau_matrix
            matrices_list.append([rescaled_tau_matrix, np.abs(r_matrix)])
        
        elif method == "TE":
            num_vars = max(df['Source'].nunique(), df['Destination'].nunique())
            te_matrix = np.zeros((num_vars, num_vars))
            p_value_matrix = np.zeros((num_vars, num_vars))
            for _, row in df.iterrows():
                source, destination = int(row['Source']) - 1, int(row['Destination']) - 1
                te_matrix[source, destination] = row['TE_Kraskov']
                p_value_matrix[source, destination] = row['P-value']
            masked_te_matrix = np.ma.masked_where(p_value_matrix > pvalue, te_matrix)
            max_te = np.max(np.abs(masked_te_matrix))
            rescaled_masked_te_matrix = np.abs(masked_te_matrix / max_te) if max_te > 0 else masked_te_matrix
            matrices_list.append([rescaled_masked_te_matrix])
    
    return matrices_list

def plot_comparison(matrices_LKIF, matrices_TE, labels, averaging_values):
    num_rows = 3  # LKIF Tau, TE, and R
    num_cols = len(averaging_values)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    fig.suptitle("Comparison of Methods for Different Averaging Parameters", fontsize=14)
    
    for col, value in enumerate(averaging_values):
        for row, (matrices, label) in enumerate(zip([matrices_LKIF, matrices_TE], ["LKIF", "TE"])):
            ax = axs[row, col]
            cax = ax.imshow(matrices[col][0], cmap="PiYG", norm=Normalize(vmin=-1, vmax=1), aspect="auto")
            ax.set_title(f"{label} - {value} days", fontsize=12)
            ax.set_xlabel("Target Variable", fontsize=10)
            ax.set_ylabel("Source Variable", fontsize=10)
            fig.colorbar(cax, ax=ax, label=label)
        ax = axs[2, col]
        cax = ax.imshow(matrices_LKIF[col][1], cmap="PiYG", norm=Normalize(vmin=-1, vmax=1), aspect="auto")
        ax.set_title(f"Correlation - {value} days", fontsize=12)
        ax.set_xlabel("Target Variable", fontsize=10)
        ax.set_ylabel("Source Variable", fontsize=10)
        fig.colorbar(cax, ax=ax, label="Correlation")
    
    plt.tight_layout()
    plt.show()

# File names for different averaging values
averaging_values = [0, 100, 1000]
file_names_LKIF = [f"../averaging/results_averaging_atmosphere/liang_res_11days_{value}days_weak_avg.csv" for value in averaging_values]
file_names_TE = [f"data_TE/results_weak_{value}days.csv" for value in averaging_values]

# Load data
matrices_LKIF = load_data(file_names_LKIF, "LKIF")
matrices_TE = load_data(file_names_TE, "TE", pvalue=0.01)

# Plot comparisons
plot_comparison(matrices_LKIF, matrices_TE, ["normalized $\\tau$", "normalized TE", "R"], averaging_values)
