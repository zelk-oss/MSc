import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors

# Global rcParams for font sizes and figure size (A4 portrait)
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 12,
    'figure.figsize': (6,10),  # A4 size in inches (portrait)
    'legend.loc' : 'upper right'
})

def plot_csv_data(csv_file):
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    i_values = data[:, 0].astype(int)  # Ensure i_values are integers for indexing
    te_values = np.abs(data[:, 1])
    error_values = data[:, 2]
    
    # Choose vector size based on max i_values
    if np.max(i_values) > 150:
        vector_days = np.logspace(1.0, 4.2, 200)  # 200 points from 10^1 to 10^4.2
    else:
        vector_days = np.logspace(1.0, 4.2, 101)
    
    # Map i_values to vector_days and convert to months
    x_values = vector_days[i_values] / (365.24/12)
    
    return x_values, te_values, error_values

def plot_four_dual(file_pairs, titles, save_path="/home/chiaraz/thesis/pictures_thesis/final/methods_comparison/batch/"):
    # Create 4 subplots arranged vertically (4 rows) with high resolution (dpi=300)
    fig, axes = plt.subplots(4, 1, figsize=(6,10), dpi=300)
    
    for ax, (file_pos, file_neg), title in zip(axes, file_pairs, titles):
        # Process the positive file
        x_pos, te_pos, err_pos = plot_csv_data(file_pos)
        # Process the negative file and flip its x-values
        x_neg, te_neg, err_neg = plot_csv_data(file_neg)
        x_neg = -x_neg  # Place these points on the negative x-axis
        
        color = 'orangered'
        # Plot both sets in blue with identical markers and labels
        ax.errorbar(x_pos, te_pos, yerr=err_pos, fmt='o', capsize=3, elinewidth=1,
                    color=color, ecolor = mcolors.to_rgba(color, alpha=0.3), markersize=2, label='LKIF')
        ax.errorbar(x_neg, te_neg, yerr=err_neg, fmt='o', capsize=3, elinewidth=1,
                    color=color, ecolor = mcolors.to_rgba(color, alpha=0.3), markersize=2, label='LKIF')
        
        # Use a symmetric log scale with adjusted parameters to reduce the gap
        ax.set_xscale('symlog')
        ax.set_xlabel("atmosphere lag wrt ocean (months)")
        ax.set_ylabel("LKIF (nats)")
        ax.set_title(title)
        
        # Deduplicate legend entries so "transfer entropy" appears only once
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    plt.subplots_adjust(hspace=.7)
    
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "lag.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    #plt.show()

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
