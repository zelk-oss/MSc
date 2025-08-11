"""
Plot the trend of TE with k = number of nearest neighbors 
for different values of mu
"""

import matplotlib.pyplot as plt
import numpy as np
import re

def load_te_file(filename):
    data = np.loadtxt(filename, skiprows=1)
    k = data[:, 0]
    te_1to2 = data[:, 1]
    std_1to2 = data[:, 2]
    te_2to1 = data[:, 4]
    std_2to1 = data[:, 5]
    return k, te_1to2, std_1to2, te_2to1, std_2to1

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

def plot_te_subplots(file_pairs, plot_alg1=True, plot_alg2=False):
    n = len(file_pairs)
    fig, axes = plt.subplots(n, 1, figsize=(6, 3 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (file1, file2) in zip(axes, file_pairs):
        mu = extract_mu(file1)
        mu_label = f"{mu:.6g}"

        if plot_alg1:
            k1, te12_1, std12_1, te21_1, std21_1 = load_te_file(file1)
            ax.errorbar(k1, te12_1, yerr=std12_1, fmt='-o',
                        label=f"ALG1 1→2, μ={mu_label}", capsize=4)
            ax.errorbar(k1, te21_1, yerr=std21_1, fmt='-o',
                        label=f"ALG1 2→1, μ={mu_label}", capsize=4)

        if plot_alg2:
            k2, te12_2, std12_2, te21_2, std21_2 = load_te_file(file2)
            ax.errorbar(k2, te12_2, yerr=std12_2, fmt='--s',
                        label=f"ALG2 1→2, μ={mu_label}", capsize=4)
            ax.errorbar(k2, te21_2, yerr=std21_2, fmt='--s',
                        label=f"ALG2 2→1, μ={mu_label}", capsize=4)

        ax.axhline(0, color='black', linestyle=':')
        ax.set_title(f"μ = {mu_label}")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Number of nearest neighbours (k)")
    axes[0].set_ylabel("Transfer Entropy (nats)")
    plt.suptitle("Transfer Entropy vs k for different μ values", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("transfer_entropy_subplots_vertical.png")
    plt.show()

if __name__ == "__main__":
    files = [
        ("te_mu0_results_alg1.txt",   "te_mu0_results_alg2.txt"),
        ("te_mu001_results_alg1.txt", "te_mu001_results_alg2.txt"),
        ("te_mu1_results_alg1.txt",   "te_mu1_results_alg2.txt")

    ]
    plot_te_subplots(files, plot_alg1=False, plot_alg2=True)

