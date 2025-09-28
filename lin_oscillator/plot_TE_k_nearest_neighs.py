"""
Plot the trend of TE with k = number of nearest neighbors 
for different values of mu
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import pickle

def load_te_file(filename):
    data = np.loadtxt(filename, skiprows=1)
    k = data[:, 0]
    te_1to2 = data[:, 1]
    std_1to2 = data[:, 2]
    pval_1to2 = data[:, 3]  # p-values 1→2
    te_2to1 = data[:, 4]
    std_2to1 = data[:, 5]
    pval_2to1 = data[:, 6]  # p-values 2→1
    return k, te_1to2, std_1to2, pval_1to2, te_2to1, std_2to1, pval_2to1

def extract_mu(filename):
    m = re.search(r'mu([0-9]+(?:\.[0-9]+)?)', filename)
    if not m:
        raise ValueError(f"Could not find mu in filename: {filename}")
    raw = m.group(1)
    if '.' in raw:
        return float(raw)
    if raw.startswith('0') and len(raw) > 1:
        return int(raw) / (10 ** len(raw))
    return float(raw)

def plot_curve_with_faded_nonsignif(ax, k, te, std, pvals, fmt, label):
    # Plot all points normally
    main_plot = ax.errorbar(k, te, yerr=std, fmt=fmt, label=label, capsize=4)

    # Extract the color used by the main plot
    main_color = main_plot[0].get_color()

    # Overlay non-significant points with lower alpha
    mask = pvals > 0.05
    if np.any(mask):
        ax.errorbar(k[mask], te[mask], yerr=std[mask], fmt='o',
                    color="black", alpha=1, capsize=4, label=None)

def plot_te_subplots(file_pairs, plot_alg1=True, plot_alg2=False):
    n = len(file_pairs)
    fig, axes = plt.subplots(n, 1, figsize=(6, 3 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (file1, file2) in zip(axes, file_pairs):
        mu = extract_mu(file1)
        mu_label = f"{mu:.6g}"

        if plot_alg1:
            k1, te12_1, std12_1, p12_1, te21_1, std21_1, p21_1 = load_te_file(file1)
            plot_curve_with_faded_nonsignif(ax, k1, te12_1, std12_1, p12_1, '-o', f"ALG 1, 1→2, μ={mu_label}")
            plot_curve_with_faded_nonsignif(ax, k1, te21_1, std21_1, p21_1, '-o', f"ALG 1, 2→1, μ={mu_label}")

        if plot_alg2:
            k2, te12_2, std12_2, p12_2, te21_2, std21_2, p21_2 = load_te_file(file2)
            plot_curve_with_faded_nonsignif(ax, k2, te12_2, std12_2, p12_2, '--s', f"ALG 2, 1→2, μ={mu_label}")
            plot_curve_with_faded_nonsignif(ax, k2, te21_2, std21_2, p21_2, '--s', f"ALG 2, 2→1, μ={mu_label}")

        ax.axhline(0, color='black', linestyle=':')
        ax.set_title(f"μ = {mu_label}")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Number of nearest neighbours (k)")
    axes[0].set_ylabel("Transfer Entropy (nats)")
    plt.suptitle("Transfer Entropy vs k for different μ values", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    #plt.savefig("transfer_entropy_subplots_vertical.png")
    if (plot_alg1 == True) and (plot_alg2 == False): 
        pickle.dump(fig, open('plot_TE_k_alg1.pickle', 'wb'))
    elif (plot_alg1 == False) and (plot_alg2 == True): 
        pickle.dump(fig, open('plot_TE_k_alg2.pickle', 'wb'))
    else: 
        print("Bad move, both algos on same canvas")
    #plt.show()

if __name__ == "__main__":
    files = [
        ("te_mu0_results_alg1.txt",   "te_mu0_results_alg2.txt"),
        ("te_mu001_results_alg1.txt", "te_mu001_results_alg2.txt"),
        ("te_mu1_results_alg1.txt",   "te_mu1_results_alg2.txt")
    ]
    plot_te_subplots(files, plot_alg1=True, plot_alg2=False)
    plot_te_subplots(files, False, True)

