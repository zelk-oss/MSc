import numpy as np
import matplotlib.pyplot as plt

# Set dimensions for the plot
rows, cols = 36, 36

# Generate random values for the variables
data = np.random.rand(rows, cols)

# Set sparse ticks for visualization
col_positions = np.arange(cols)  # Column indices
row_positions = np.arange(rows)  # Row indices

# Adjust tick sparseness
n_cols = 5  # Show every nth column tick
n_rows = 5  # Show every nth row tick

sparse_col_positions = col_positions[::n_cols]
sparse_row_positions = row_positions[::n_rows]

sparse_col_labels = [f'C{i+1}' for i in sparse_col_positions]  # Example column labels
sparse_row_labels = [f'R{i+1}' for i in sparse_row_positions]  # Example row labels

# Create the plot
plt.figure(figsize=(10, 10))  # Set figure size for better readability

# Use a heatmap to visualize data
plt.imshow(data, cmap='viridis', aspect='auto')

# Customize ticks and labels
plt.xticks(ticks=sparse_col_positions, labels=sparse_col_labels, rotation=45)
plt.yticks(ticks=sparse_row_positions, labels=sparse_row_labels)

# Add labels and title
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.title('26x36 Random Variable Heatmap with Sparse Ticks')

# Add a colorbar for the heatmap
plt.colorbar(label='Random Value Intensity')

# Display the plot
plt.tight_layout()
plt.show()
