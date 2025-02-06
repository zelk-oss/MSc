import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize

def load_TE_data(filename, pvalue):
    """Load and process TE data from CSV files."""
    data = pd.read_csv(filename)
    data.columns = data.columns.str.strip()
    
    sources = data['Source'].unique()
    destinations = data['Destination'].unique()
    num_vars = max(len(sources), len(destinations))
    
    te_matrix = np.zeros((num_vars, num_vars))
    p_value_matrix = np.zeros((num_vars, num_vars))
    
    for _, row in data.iterrows():
        source = int(row['Source']) - 1  # Convert 1-based to 0-based
        destination = int(row['Destination']) - 1
        te_value = row['TE_Kraskov']
        p_value = row['P-value']
        te_matrix[source, destination] = te_value
        p_value_matrix[source, destination] = p_value
    
    masked_te_matrix = np.ma.masked_where(p_value_matrix > pvalue, te_matrix)
    max_te = np.max(np.abs(masked_te_matrix))
    rescaled_masked_te_matrix = masked_te_matrix / max_te if max_te > 0 else masked_te_matrix
    
    return rescaled_masked_te_matrix

def plot_TE_matrices(file_names, title, pvalue=0.01):
    """Plot TE matrices for the given file names."""
    te_matrices = {label: load_TE_data(filename, pvalue) for label, filename in file_names.items()}
    
    fig, axs = plt.subplots(1, len(te_matrices), figsize=(5 * len(te_matrices), 5))
    fig.suptitle(title)
    
    for ax, (label, matrix) in zip(axs, te_matrices.items()):
        cax = ax.imshow(matrix, cmap="PiYG", norm=Normalize(vmin=-1, vmax=1), aspect="auto")
        ax.set_title(label)
        ax.set_xlabel("Target Variable")
        ax.set_ylabel("Source Variable")
        
        num_vars = matrix.shape[0]
        n = max(1, num_vars // 5)
        ticks = np.arange(0, num_vars, n)
        tick_labels = np.arange(1, num_vars + 1, n)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        
        fig.colorbar(cax, ax=ax, label="Normalized TE")
    
    plt.tight_layout()
    plt.show()

def compare_strong_weak(file_names_weak, file_names_strong, pvalue=0.01):
    """Compare TE values between strong and weak coupling conditions."""
    te_matrices_weak = {label: load_TE_data(filename, pvalue) for label, filename in file_names_weak.items()}
    te_matrices_strong = {label: load_TE_data(filename, pvalue) for label, filename in file_names_strong.items()}
    
    fig, axs = plt.subplots(2, len(te_matrices_weak), figsize=(5 * len(te_matrices_weak), 10))
    fig.suptitle("Comparison of TE values - Strong vs Weak Coupling")
    
    for i, (label, matrix) in enumerate(te_matrices_weak.items()):
        cax = axs[0, i].imshow(matrix, cmap="PiYG", norm=Normalize(vmin=-1, vmax=1), aspect="auto")
        axs[0, i].set_title(f"Weak Coupling - {label}")
        axs[0, i].set_xlabel("Target Variable")
        axs[0, i].set_ylabel("Source Variable")
        fig.colorbar(cax, ax=axs[0, i], label="Normalized TE")
        
    for i, (label, matrix) in enumerate(te_matrices_strong.items()):
        cax = axs[1, i].imshow(matrix, cmap="PiYG", norm=Normalize(vmin=-1, vmax=1), aspect="auto")
        axs[1, i].set_title(f"Strong Coupling - {label}")
        axs[1, i].set_xlabel("Target Variable")
        axs[1, i].set_ylabel("Source Variable")
        fig.colorbar(cax, ax=axs[1, i], label="Normalized TE")
    
    plt.tight_layout()
    plt.show()

# Example usage:
file_names_weak = {
    "Original": "data_TE/results_weak_largewindow.csv",
    "1 year running mean": "data_TE/results_1yr_weak_largewindow.csv",
    "10 years running mean": "data_TE/results_10yr_weak_largewindow.csv",
    "100 years running mean": "data_TE/results_100yr_weak_largewindow.csv"
}

file_names_strong = {
    "Original": "data_TE/results_strong_largewindow.csv",
    "1 year running mean": "data_TE/results_1yr_strong_largewindow.csv",
    "10 years running mean": "data_TE/results_10yr_strong_largewindow.csv",
    "100 years running mean": "data_TE/results_100yr_strong_largewindow.csv"
}

# Plot weak or strong TE matrices
plot_TE_matrices(file_names_weak, "Weak Coupling TE Matrices")
plot_TE_matrices(file_names_strong, "Strong Coupling TE Matrices")

# Compare strong vs weak TE matrices
compare_strong_weak(file_names_weak, file_names_strong, pvalue=0.01)
