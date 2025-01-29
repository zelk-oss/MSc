import matplotlib.pyplot as plt
import numpy as np

# okay questo plot va ripensato 

# Data for the plots
conditions = ["Forte", "Debole"]
values_TAB = [-0.01478456923242677, -0.02602649630898224]
values_TBA = [1.8055582999289328, 0.02963900814905074]
significant_TAB = [False, True]
significant_TBA = [True, False]

# Prepare the matrix and colors
matrix = np.array([[values_TAB[0], values_TBA[0]], [values_TAB[1], values_TBA[1]]])
color_matrix = [
    ["white" if not significant_TAB[0] else plt.cm.viridis(0.5), "white" if not significant_TBA[0] else plt.cm.viridis(0.8)],
    ["white" if not significant_TAB[1] else plt.cm.viridis(0.5), "white" if not significant_TBA[1] else plt.cm.viridis(0.8)]
]

# Plot setup
fig, ax = plt.subplots(figsize=(6, 3))
ax.set_xticks([0, 1])
ax.set_xticklabels(["TAB", "TBA"])
ax.set_yticks([0, 1])
ax.set_yticklabels(conditions)

# Add the colored boxes
for i in range(2):
    for j in range(2):
        ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, color=color_matrix[i][j]))
        text = f"{matrix[i, j]:.2f}"
        ax.text(j + 0.5, 1 - i + 0.5, text, ha="center", va="center", fontsize=12)

# Adjust layout
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_aspect("equal")
ax.set_title("Comparison of TAB and TBA", fontsize=14)
plt.tight_layout()

# Show the plot
plt.show()
