import matplotlib.pyplot as plt
import numpy as np

def load_te_file(filename):
    data = np.loadtxt(filename, skiprows=1)
    k = data[:, 0]
    te_1to2 = data[:, 1]
    std_1to2 = data[:, 2]
    pval_1to2 = data[:, 3]  # ✅ assuming p-value is here
    te_2to1 = data[:, 4]
    std_2to1 = data[:, 5]
    pval_2to1 = data[:, 6]  # ✅ assuming p-value is here
    return k, te_1to2, std_1to2, pval_1to2, te_2to1, std_2to1, pval_2to1

def plot_te_results():
    k1, te12_alg1, std12_alg1, pval12_alg1, te21_alg1, std21_alg1, pval21_alg1 = load_te_file("te_results_alg1.txt")
    k2, te12_alg2, std12_alg2, pval12_alg2, te21_alg2, std21_alg2, pval21_alg2 = load_te_file("te_results_alg2.txt")

    plt.figure(figsize=(10, 6))
    """
    # --- ALG 1 ---
    # Plot significant points with a line
    plt.errorbar(k1, te12_alg1, yerr=std12_alg1, fmt='-o', label="TE 1→2 (ALG 1)", capsize=4)
    plt.errorbar(k1, te21_alg1, yerr=std21_alg1, fmt='-o', label="TE 2→1 (ALG 1)", capsize=4)

    # Overlay non-significant points in black (no line)
    mask_ns_12 = pval12_alg1 > 0.05
    if np.any(mask_ns_12):
        plt.errorbar(k1[mask_ns_12], te12_alg1[mask_ns_12], yerr=std12_alg1[mask_ns_12],
                     fmt='o', color='black', capsize=4, label=None)

    mask_ns_21 = pval21_alg1 > 0.05
    if np.any(mask_ns_21):
        plt.errorbar(k1[mask_ns_21], te21_alg1[mask_ns_21], yerr=std21_alg1[mask_ns_21],
                     fmt='o', color='black', capsize=4, label=None)
    """
    # --- ALG 2 ---
    # (Uncomment if you want ALG 2 as well)
    plt.errorbar(k2, te12_alg2, yerr=std12_alg2, fmt='--s', label="TE 1→2 (ALG 2)", capsize=4)
    plt.errorbar(k2, te21_alg2, yerr=std21_alg2, fmt='--s', label="TE 2→1 (ALG 2)", capsize=4)

    mask_ns_12 = pval12_alg2 > 0.05
    if np.any(mask_ns_12):
        plt.errorbar(k2[mask_ns_12], te12_alg2[mask_ns_12], yerr=std12_alg2[mask_ns_12],
                     fmt='o', color='black', capsize=4, label=None)

    mask_ns_21 = pval21_alg2 > 0.05
    if np.any(mask_ns_21):
        plt.errorbar(k2[mask_ns_21], te21_alg2[mask_ns_21], yerr=std21_alg2[mask_ns_21],
                     fmt='o', color='black', capsize=4, label=None)

    # TE = 0 line
    plt.axhline(0, color='black', linestyle=':', label="TE = 0")

    plt.xlabel("Number of nearest neighbours (k)")
    plt.ylabel("Transfer Entropy (nats)")
    plt.title("Transfer Entropy vs k (ALG 1 & 2)\nColumns 1↔2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("transfer_entropy_vs_k.png")
    plt.show()

if __name__ == "__main__":
    plot_te_results()
