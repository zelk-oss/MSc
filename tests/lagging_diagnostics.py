# just a script to test the function 

import numpy as np
import matplotlib.pyplot as plt

def introduce_lag_diagnostics(data1, data2, months_of_delay, points_per_month, apply_lag):
    """
    Introduces a lag between two 3D datasets in the format [time, x, y] and aligns them after applying the lag.

    Parameters:
        data1 (np.ndarray): The first dataset (time, x, y).
        data2 (np.ndarray): The second dataset (time, x, y).
        months_of_delay (float): Delay in months (positive = data2 lags backward, negative = data1 lags backward).
        points_per_month (float): Conversion factor for time points per month.
        apply_lag (bool): Whether to apply the lag.

    Returns:
        tuple: A tuple containing the two datasets after applying the lag and trimming.
    """
    if not apply_lag or months_of_delay == 0:
        return data1, data2  # Return the original data if no lag is applied

    # Convert months of delay to time points
    lag_points = int(months_of_delay * points_per_month)

    # Shape of the data: (time, x, y)
    time, x, y = data1.shape

    if lag_points > 0:
        # data2 lags behind data1
        trimmed_data1 = data1[:-lag_points, :, :]  # Trim the first dataset
        trimmed_data2 = data2[lag_points:, :, :]  # Trim the second dataset
    else:
        # data1 lags behind data2
        lag_points = abs(lag_points)
        trimmed_data1 = data1[lag_points:, :, :]  # Trim the first dataset
        trimmed_data2 = data2[:-lag_points, :, :]  # Trim the second dataset

    # Align the time dimensions after trimming
    min_time = min(trimmed_data1.shape[0], trimmed_data2.shape[0])
    trimmed_data1 = trimmed_data1[:min_time, :, :]
    trimmed_data2 = trimmed_data2[:min_time, :, :]

    print(f"Lag applied: {months_of_delay} months ({lag_points} time points)")
    return trimmed_data1, trimmed_data2

# Example usage:
ntime = 100  # Number of time steps
nx, ny = 10, 10  # Spatial dimensions (x, y)

# Create two 3D datasets with a step function pattern
data1 = np.zeros((ntime, nx, ny))
data2 = np.zeros((ntime, nx, ny))

# Apply the step function: 
# data1 will be 1 for the first half and 0 for the second half
data1[ntime // 2:, :, :] = 1

# data2 will be 0 for the first half and 1 for the second half
data2[:ntime // 2, :, :] = 0
data2[ntime // 2:, :, :] = 1

# Apply a lag
months_of_delay = 20  # 10 months lag between the two datasets
points_per_month = 1
apply_lag = True

lagged_data1, lagged_data2 = introduce_lag_3d(data1, data2, months_of_delay, points_per_month, apply_lag)

# Visualize the result (showing one of the spatial points for simplicity)
plt.figure(figsize=(12, 6))

# Plot the first spatial point (e.g., [0, 0]) for both datasets before and after the lag
plt.subplot(1, 2, 1)
plt.title("Before Lag: Dataset 1 vs Dataset 2 at [0,0]")
plt.plot(data1[:, 0, 0], label="Dataset 1 (Atmosphere)", color='red', alpha=0.7)
plt.plot(data2[:, 0, 0], label="Dataset 2 (Ocean)", color='blue', alpha=0.7)
plt.xlabel("Time Points")
plt.ylabel("Value")
plt.legend()

# Plot the first spatial point after the lag
plt.subplot(1, 2, 2)
plt.title(f"After Lag: Dataset 1 vs Dataset 2 at [0,0] (Lag = {months_of_delay} months)")
plt.plot(lagged_data1[:, 0, 0], label="Lagged Dataset 1 (Atmosphere)", color='red', alpha=0.7)
plt.plot(lagged_data2[:, 0, 0], label="Lagged Dataset 2 (Ocean)", color='blue', alpha=0.7)
plt.xlabel("Time Points")
plt.ylabel("Value")
plt.legend()

plt.tight_layout()
plt.show()
