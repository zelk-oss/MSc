import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter

# Global rcParams for font sizes and figure size (A4 portrait)
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 12,
    'figure.figsize': (9, 13)  # A4 size in inches (portrait)
})

def plot_csv_data(csv_file, plot_scale="log"):
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    data = data[::2,:] # subsample
    i_values = data[:, 0].astype(int)  # Ensure i_values are integers for indexing
    te_values = np.abs(data[:, 1])
    error_values = data[:, 2]
    
    # Choose vector size based on max i_values
    if np.max(i_values) > 150:
        vector_days = np.logspace(1.0, 4.2, 200)  # 200 points from 10^1 to 10^4.2
    else:
        vector_days = np.logspace(1.0, 4.2, 101)
    
    # Map i_values to vector_days and convert to years
    x_values = vector_days[i_values] / (365.24/12)
    
    return x_values, te_values, error_values

def plot_four(file_list, titles, save_path="/home/chiaraz/thesis/pictures_thesis/final/methods_comparison/batch/"):
    # Create 4 subplots in one column (4 rows) with high resolution (dpi=300)
    fig, axes = plt.subplots(4, 1, figsize=(8.27, 11.69), dpi=300)
    scales = ["log"] * len(file_list)
    
    for ax, scale, csv_file, title in zip(axes, scales, file_list, titles):
        x_values, te_values, error_values = plot_csv_data(csv_file, plot_scale=scale)
        ax.errorbar(x_values, te_values, yerr=error_values, fmt='o', capsize=3, elinewidth=1,
                    color='blue', ecolor=(0, 0, 1, 0.3), markersize=4, label='transfer entropy')
        ax.set_xscale(scale)
        ax.set_xlabel("running mean window size (months)")
        ax.set_ylabel("transfer entropy")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Increase vertical spacing between subplots
    plt.subplots_adjust(hspace=0.7)
    
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "LKIF_average_plots.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    #plt.show()

# Example usage with a list of CSV file paths and corresponding titles
files = [
    "output_LKIF_strong_atmosphere_ocean.csv",
    "output_LKIF_strong_ocean_atmosphere.csv",
    "output_LKIF_weak_atmosphere_ocean.csv",
    "output_LKIF_strong_ocean_atmosphere.csv"
]
titles = [
    r"Atmosphere $\to$ Ocean - $d = 1.1e^{-7}$",
    r"Ocean $\to$ Atmosphere - $d = 1.1e^{-7}$",
    r"Atmosphere $\to$ Ocean - $d = 1.e^{-8}$",
    r"Ocean $\to$ Atmosphere - $d = 1.e^{-8}$"
]

plot_four(files, titles)
