# plot_ais.py

import matplotlib.pyplot as plt
import numpy as np

def plot_ais(file_path="ais_results_mu0.01.txt"):

    data = np.loadtxt(file_path, skiprows=1)
    k = data[:, 0]
    ais_col1 = data[:, 1]
    ais_col2 = data[:, 2]

    plt.figure(figsize=(8, 5))
    plt.plot(k, ais_col1, marker='o', label="Column 1")
    plt.plot(k, ais_col2, marker='s', label="Column 2")

    plt.title(f"Active Information Storage vs k_history. mu = {mu}")
    plt.xlabel("k_history")
    plt.ylabel("AIS (nats)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ais_vs_k_history.png")
    plt.show()

if __name__ == "__main__":
    plot_ais()
