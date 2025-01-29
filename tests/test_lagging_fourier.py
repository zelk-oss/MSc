import numpy as np 
import matplotlib.pyplot as plt 

import sys
sys.path.insert(0, '/home/chiaraz/thesis')
from functions_for_maooam import introduce_lag_fourier

# Create a time series dataset (36 variables x 100 time points)
nvar = 36
ntime = 1000
points_per_day = 1/11  # 1 point per month (could be different for higher resolution)

# Create a step function: 1 for first half, 0 for second half for all variables
data = np.ones((nvar, ntime))
data[:, ntime//2:] = 0  # Set the second half of the data to 0

# Apply a lag
days_of_delay = 365.25 / 2  # 6 months lag between atmosphere and ocean

lagged_data = introduce_lag_fourier(data, days_of_delay, True)

# Visualize the result
# Plot the first component of the atmosphere and ocean before and after lag
plt.figure(figsize=(12, 6))

# Plot the first atmospheric and oceanic variable before the lag
plt.subplot(1, 2, 1)
plt.title("Before Lag: First Atmospheric vs. Oceanic Variable")
plt.plot(data[0, :], label="Atmospheric Component", color='red', alpha=0.7)
plt.plot(data[21, :], label="Oceanic Component", color='blue', alpha=0.7)
plt.xlabel("Time Points")
plt.ylabel("Value")
plt.legend()

print(lagged_data)

# Plot the first atmospheric and oceanic variable after the lag
plt.subplot(1, 2, 2)
plt.title(f"After Lag: First Atmospheric vs. Oceanic Variable (Lag = {days_of_delay} days)")
plt.plot(lagged_data[0, :], label="Lagged Atmospheric Component", color='red', alpha=0.7)
plt.plot(lagged_data[20, :], label="Lagged Oceanic Component", color='blue', alpha=0.7)
plt.xlabel("Time Points")
plt.ylabel("Value")
plt.legend()

plt.tight_layout()
plt.show()
