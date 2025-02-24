import numpy as np
import pandas as pd
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
    'figure.figsize': (9, 13)
})

def plot_csv_data_lkif_dual(csv_file):
    """
    Reads a LKIF data file with the columns:
      File, TAB, TBA, Error TAB, Error TBA, Significant TAB, Significant TBA

    We only read the first 5 columns (ignoring the Significant ones) because 
    extra commas in those fields can cause column-splitting errors.
    
    The file contains both flows. We split the data into:
      - Positive lags: rows where the File field does NOT contain 'neg'. 
        For these, we use the TAB column (atmosphere → ocean) and its error.
      - Negative lags: rows where the File field DOES contain 'neg'. 
        For these, we use the TBA column (ocean → atmosphere) and its error.
    
    The error is scaled to a 95% confidence interval by multiplying by 1.96.
    Data is subsampled (every 2nd row).
    
    Returns:
      x_pos, lkif_values_pos, error95_pos, x_neg, lkif_values_neg, error95_neg
    """
    # Read only columns 0-4 using pandas
    df = pd.read_csv(csv_file, delimiter=',', usecols=[0, 1, 2, 3, 4], header=0)
    # Rename columns to remove spaces (e.g. "Error TAB" -> "Error_TAB")
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    # Subsample every 2nd row
    df = df.iloc[::2, :]
    
    # Split data into positive and negative lags based on the 'File' column
    df['File'] = df['File'].astype(str)
    pos_mask = ~df['File'].str.contains('neg')
    neg_mask = df['File'].str.contains('neg')
    df_pos = df[pos_mask]
    df_neg = df[neg_mask]
    
    # For positive data: use TAB and Error_TAB.
    # Remove trailing 's' or 'w' from the File field and convert to int.
    i_values_pos = np.array([int(x.rstrip('sw')) for x in df_pos['File']], dtype=np.int64)
    lkif_values_pos = df_pos['TAB'].to_numpy()
    error_values_pos = df_pos['Error_TAB'].to_numpy()
    error95_pos = 1.96 * error_values_pos

    # For negative data: use TBA and Error_TBA.
    # Remove the leading "neg" and trailing 's' or 'w'
    i_values_neg = np.array([int(x.lstrip('neg').rstrip('sw')) for x in df_neg['File']], dtype=np.int64)
    lkif_values_neg = df_neg['TBA'].to_numpy()
    error_values_neg = df_neg['Error_TBA'].to_numpy()
    error95_neg = 1.96 * error_values_neg

    # Determine the maximum index to build the lag vector.
    max_index = 0
    if i_values_pos.size > 0:
        max_index = max(max_index, np.max(i_values_pos))
    if i_values_neg.size > 0:
        max_index = max(max_index, np.max(i_values_neg))
    
    # Use a logarithmically spaced vector; if max_index is large, use 200 points; otherwise, 101.
    if max_index > 150:
        vector_days = np.logspace(1.0, 4.2, 200)
    else:
        vector_days = np.logspace(1.0, 4.2, 101)
    
    # Map indices to x-values in months
    x_pos = vector_days[i_values_pos] / (365.24/12)
    x_neg = vector_days[i_values_neg] / (365.24/12)
    
    return x_pos, lkif_values_pos, error95_pos, x_neg, lkif_values_neg, error95_neg

def plot_four_lkif_scenarios(file_list, titles, save_path="/home/chiaraz/thesis/pictures_thesis/final/methods_comparison/batch/"):
    """
    Expects file_list as a list of 2 LKIF data files:
      - file_list[0]: strong coupling (e.g., d = 1.1e-7)
      - file_list[1]: weak coupling (e.g., d = 1.e-8)
    Each file contains both flows.

    This function creates a 2×2 grid of subplots:
      Top row: Atmosphere → Ocean (positive lags, using TAB)
      Bottom row: Ocean → Atmosphere (negative lags, using TBA with x-values flipped)
      Left column: strong coupling; right column: weak coupling.
    
    The titles argument is a list of 4 strings in the order:
      [0]: Atmosphere → Ocean (strong)
      [1]: Ocean → Atmosphere (strong)
      [2]: Atmosphere → Ocean (weak)
      [3]: Ocean → Atmosphere (weak)
    """
    if len(file_list) != 2:
        raise ValueError("file_list must contain exactly 2 files: [strong_file, weak_file].")
    
    fig, axes = plt.subplots(2, 2, figsize=(9, 13), dpi=300)
    
    # Loop over the two files (strong and weak coupling)
    for i, csv_file in enumerate(file_list):
        x_pos, lkif_pos, err_pos, x_neg, lkif_neg, err_neg = plot_csv_data_lkif_dual(csv_file)
        
        # Top row: Atmosphere → Ocean (positive lags)
        ax_top = axes[0, i]
        ax_top.errorbar(x_pos, lkif_pos, yerr=err_pos, fmt='o', capsize=3, elinewidth=1,
                        color='orangered', ecolor=(1, 0.27, 0, 0.3), markersize=4, label='LKIF')
        ax_top.set_xscale('symlog')
        ax_top.set_xlabel("lag (months)")
        ax_top.set_ylabel("LKIF")
        ax_top.set_title(titles[0] if i == 0 else titles[2])
        ax_top.legend()
        ax_top.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax_top.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # Bottom row: Ocean → Atmosphere (negative lags)
        ax_bot = axes[1, i]
        # Flip the x-values so that they appear on the negative side of zero.
        ax_bot.errorbar(-x_neg, lkif_neg, yerr=err_neg, fmt='o', capsize=3, elinewidth=1,
                        color='orangered', ecolor=(1, 0.27, 0, 0.3), markersize=4, label='LKIF')
        ax_bot.set_xscale('symlog')
        ax_bot.set_xlabel("lag (months)")
        ax_bot.set_ylabel("LKIF")
        ax_bot.set_title(titles[1] if i == 0 else titles[3])
        ax_bot.legend()
        ax_bot.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax_bot.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "lkif_scenarios.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    #plt.show()

# Example usage:
file_list_lkif = [
    "log_results_strong.csv",  # Strong coupling (d = 1.1e-7)
    "log_results_weak.csv"     # Weak coupling (d = 1.e-8)
]

titles_lkif = [
    r"Atmosphere $\to$ Ocean - $d = 1.1e^{-7}$",  # Strong, atmo→ocean (top left)
    r"Ocean $\to$ Atmosphere - $d = 1.1e^{-7}$",    # Strong, ocean→atmo (bottom left)
    r"Atmosphere $\to$ Ocean - $d = 1.e^{-8}$",      # Weak, atmo→ocean (top right)
    r"Ocean $\to$ Atmosphere - $d = 1.e^{-8}$"       # Weak, ocean→atmo (bottom right)
]

plot_four_lkif_scenarios(file_list_lkif, titles_lkif)
