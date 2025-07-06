"""In this script we integrate a simple system
already used in _A comparison of two causal methods in the context of climate analyses_
In a later script we will compute TE and IF to check the properties of embedding dimension 
on a system with known causality and where we can cross check with published paper results"""

"""
Generate 2D time series for stochastic differential system
and save them to a text file for further analysis.

Equation from Liang (2014):
dX1 = (a11 * X1 + a12 * X2) * dt + sigma1 * dW1
dX2 = (a22 * X2 + a21 * X1) * dt + sigma2 * dW2
"""

import numpy as np
import matplotlib.pyplot as plt 

# Time parameters
dt = 0.01
tmax = 400
nt = int(tmax / dt)
t = np.linspace(0, tmax, nt)

# Equation parameters
a11 = -1
a12 = 0.5
a21 = 0
a22 = -1
sigma1 = 0.1
sigma2 = 0.1

# Noise
np.random.seed(42)  # for reproducibility
dW1 = np.sqrt(dt) * np.random.normal(0, 1, nt)
dW2 = np.sqrt(dt) * np.random.normal(0, 1, nt)

# Initialization
X1 = np.zeros(nt)
X2 = np.zeros(nt)
X1[0] = 1
X2[0] = 2

# Euler-Maruyama integration
for i in range(nt - 1):
    X1[i + 1] = X1[i] + (a11 * X1[i] + a12 * X2[i]) * dt + sigma1 * dW1[i]
    X2[i + 1] = X2[i] + (a22 * X2[i] + a21 * X1[i]) * dt + sigma2 * dW2[i]

# Save time series to text file (columns: time, X1, X2)
output_file = "/home/chiaraz/data_thesis/2D_system_data/2D_timeseries.txt"
np.savetxt(output_file, np.column_stack((t, X1, X2)), header="time X1 X2")
print(f"Time series saved to {output_file}")


# Plot the results
plt.figure(figsize=(10, 4))
plt.plot(t, X1, label='X1', alpha=0.8)
plt.plot(t, X2, label='X2', alpha=0.8)
plt.title("Integrated Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.grid(True)
#plt.show()