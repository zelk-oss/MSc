import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors

# Adjusted rcParams for better spacing and clarity
plt.rcParams.update({
    'font.size': 10,            # Base font size
    'axes.titlesize': 9,        # Subplot title size
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 8,
    'legend.fontsize': 10,
    'legend.loc': "upper left",
    'figure.figsize': (8, 6)     # Figure size (in inches)
})

def plot_csv_data(csv_file):
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    data = data[::2, :]  # Subsample
    i_values = data[:, 0].astype(int)  # Ensure integer indices
    te_values = np.abs(data[:, 1])
    error_values = data[:, 2]
    
    # Choose vector size based on max i_values
    if np.max(i_values) > 150:
        vector_days = np.logspace(1.0, 4.2, 200)
    else:
        vector_days = np.logspace(1.0, 4.2, 101)
    
    divider = (365.24 / 12)
    # Map i_values to vector_days and convert to months
    x_values = vector_days[i_values] / divider
    
    return x_values, te_values, error_values

def plot_four_dual(file_pairs, titles, save_path="/home/chiaraz/thesis/pictures_thesis/final/methods_comparison/batch/"):
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, dpi=300, figsize=(8,6))
    
    # Pre-calculate x positions for 100 and 1000 days (converted to months)
    x_100 = 100 / (365.24 / 12)    # ~3.29 months
    x_1000 = 1000 / (365.24 / 12)  # ~32.9 months

    # Loop over each subplot (flattened index corresponds to file_pairs list order)
    for ax, (file_pos, file_neg), title in zip(axes.flat, file_pairs, titles):
        # Process the positive file
        x_pos, te_pos, err_pos = plot_csv_data(file_pos)
        # Process the negative file and flip its x-values for symmetry
        x_neg, te_neg, err_neg = plot_csv_data(file_neg)
        x_neg = -x_neg
        
        color = 'orangered'
        # Plot the positive and negative curves with error bars
        ax.errorbar(x_pos, te_pos, yerr=err_pos, fmt='o', capsize=3, elinewidth=1,
                    color=color, ecolor=mcolors.to_rgba(color, alpha=0.25), markersize=1.2, label='LKIF')
        ax.errorbar(x_neg, te_neg, yerr=err_neg, fmt='o', capsize=3, elinewidth=1,
                    color=color, ecolor=mcolors.to_rgba(color, alpha=0.25), markersize=1.2, label='LKIF')
        
        # Use a symmetric log scale so negative and positive values are shown
        ax.set_xscale('symlog')
        ax.set_xlabel("atmosphere lag wrt ocean (months)", fontsize=8)
        ax.set_ylabel("LKIF (nats)", fontsize=8)
        # Keep only the directional part of the title (before the semicolon)
        subplot_title = title.split(";")[0].strip()
        ax.set_title(subplot_title, fontsize=9)
        
        # Deduplicate legend entries so "LKIF" appears only once
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # Add red vertical dashed lines for 100 days and 1000 days
        #ax.axvline(x=x_100, color='red', linestyle='--', linewidth=1)
        #ax.axvline(x=x_1000, color='red', linestyle='--', linewidth=1)
        ylim = ax.get_ylim()
        # Position the labels inside the plot at 93% and 88% of the y-limit
        #ax.text(x_100, ylim[1]*0.93, "100 days", color='red', fontsize=8, ha='center', va='top')
        #ax.text(x_1000, ylim[1]*0.80, "1000 days", color='red', fontsize=8, ha='center', va='top')
    
    # Add row titles for coupling types using fig.text in normalized figure coordinates
    fig.text(0.5, 0.92, "Strong Coupling", ha="center", va="top", fontsize=12)
    fig.text(0.5, 0.49, "Weak Coupling", ha="center", va="top", fontsize=12)
    
    # Adjust spacing between subplots and reserve extra space at top and bottom
    plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.22, hspace=0.6)
    
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "lag.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    # plt.show()

# Example usage with file pairs: each tuple contains (positive_file, negative_file)
file_pairs = [
    ("output_LKIF_strong_atmosphere_ocean.csv", "output_LKIF_NEGstrong_atmosphere_ocean.csv"),
    ("output_LKIF_strong_ocean_atmosphere.csv", "output_LKIF_NEGstrong_ocean_atmosphere.csv"),
    ("output_LKIF_weak_atmosphere_ocean.csv", "output_LKIF_NEGweak_atmosphere_ocean.csv"),
    ("output_LKIF_weak_ocean_atmosphere.csv", "output_LKIF_NEGweak_ocean_atmosphere.csv")
]

titles = [
    r"Atmosphere $\to$ Ocean; strong coupling",
    r"Ocean $\to$ Atmosphere; strong coupling",
    r"Atmosphere $\to$ Ocean; weak coupling",
    r"Ocean $\to$ Atmosphere; weak coupling"
]

plot_four_dual(file_pairs, titles)
