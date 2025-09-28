import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Regions (columns of your TE results file)
regions = ["EPAC", "CPAC", "NAT", "WPAC"]
nvar = len(regions)

# Initialize matrices
TE = np.zeros((nvar, nvar))
pvals = np.ones((nvar, nvar))

# --- Parse file (te_results.txt) ---
with open("te_results.txt", "r") as f:
    for line in f:
        match = re.search(
            r"col_(\d+) -> col_(\d+)\).*?=\s*([-\d.]+) nats.*?p\(surrogate > measured\)=([\d.]+)",
            line
        )
        if match:
            src = int(match.group(1))
            dst = int(match.group(2))
            te_val = float(match.group(3))
            pval = float(match.group(4))
            TE[src, dst] = te_val
            pvals[src, dst] = pval

# --- Normalize around 0 ---
max_val = np.nanmax(np.abs(TE))  # symmetric scaling
norm = Normalize(vmin=-max_val, vmax=max_val)

# --- Plot TE matrix ---
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(TE, cmap="coolwarm", norm=norm)

# Add text annotations (TE value + significance mark)
for i in range(nvar):
    for j in range(nvar):
        text = f"{TE[i,j]:.3f}"
        if pvals[i,j] <= 0.05 and i != j:  # use 0.05 threshold (to match title)
            text += " *"
        ax.text(j, i, text, ha="center", va="center", color="black")

# Configure axes
ax.set_xticks(range(nvar))
ax.set_yticks(range(nvar))
ax.set_xticklabels(regions)
ax.set_yticklabels(regions)
plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

ax.set_title("Transfer Entropy Matrix (nats)\n* p ≤ 0.05")
fig.colorbar(cax, label="TE value (nats)")
plt.tight_layout()
plt.show()
