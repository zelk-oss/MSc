import numpy as np
import matplotlib.pyplot as plt

# Files and labels
files = [
    ("te_results_mu0_notdownsampled.npz", "mu = 0"),
    ("te_results_mu1_notdownsampled.npz", "mu = 0.001"),
    ("te_results_mu2_notdownsampled.npz", "mu = 0.1"),
]

# Load all results first
results = []
for f, label in files:
    data = np.load(f)
    results.append((data["TE_O_to_A"], data["TE_A_to_O"], data["m_tot"], data["L_tot"], label))

# Plot grid: rows = mu values, cols = directions
fig, axes = plt.subplots(len(results), 2, figsize=(12, 4 * len(results)),
                         constrained_layout=True)
plt.title("GC TE, no downsampling, 6*10^5 data points")
if len(results) == 1:
    axes = np.array([axes])  # make sure it's always 2D

for i, (TE_O_to_A, TE_A_to_O, m_tot, L_tot, label) in enumerate(results):
    m_indices = np.arange(len(m_tot))
    L_indices = np.arange(len(L_tot))
    m_tick_pos = np.arange(0, len(m_indices), 10)
    L_tick_pos = np.arange(0, len(L_indices), 10)

    # Compute normalization for this row only
    vmin = min(np.nanmin(TE_O_to_A), np.nanmin(TE_A_to_O))
    vmax = max(np.nanmax(TE_O_to_A), np.nanmax(TE_A_to_O))

    # O → A
    im1 = axes[i, 0].pcolormesh(m_indices, L_indices, TE_O_to_A.T,
                                 shading="auto", vmin=vmin, vmax=vmax)
    axes[i, 0].set_title(f"{label}, ocean → atmosphere")
    axes[i, 0].set_xticks(m_tick_pos)
    axes[i, 0].set_xticklabels(m_tot[m_tick_pos])
    axes[i, 0].set_yticks(L_tick_pos)
    axes[i, 0].set_yticklabels(L_tot[L_tick_pos])
    axes[i, 0].set_xlabel("history (log scale)")
    axes[i, 0].set_ylabel("lag (log scale)")

    # A → O
    im2 = axes[i, 1].pcolormesh(m_indices, L_indices, TE_A_to_O.T,
                                 shading="auto", vmin=vmin, vmax=vmax)
    axes[i, 1].set_title(f"{label}, atmosphere → ocean")
    axes[i, 1].set_xticks(m_tick_pos)
    axes[i, 1].set_xticklabels(m_tot[m_tick_pos])
    axes[i, 1].set_yticks(L_tick_pos)
    axes[i, 1].set_yticklabels(L_tot[L_tick_pos])
    axes[i, 1].set_xlabel("history (log scale)")
    axes[i, 1].set_ylabel("lag (log scale)")

    # Colorbar for this row (two subplots)
    cbar = fig.colorbar(im2, ax=axes[i, :].ravel().tolist(),
                        orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("TE")

plt.savefig("results_mu.png")
