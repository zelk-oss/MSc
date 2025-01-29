import numpy as np 
import matplotlib.pyplot as plt 

def average_time_series(data, window_size, apply_averaging):
    """
    Averages the data over a manually set number of time points.

    Parameters:
        data (np.ndarray): The input time_series array (nvar x N matrix).
        window_size (int): Number of months to average over.
        apply_averaging (bool): Whether to apply averaging.

    Returns:
        np.ndarray: The averaged time_series array (or original if not applied).
    """
    if not apply_averaging:
        return data  # Return the original data if averaging is not applied

    # Compute the new length after averaging
    nvar, ntime = data.shape
    new_length = ntime // window_size

    # Initialize an array to store averaged results
    averaged_data = np.empty((nvar, new_length))
    
    # !!!!!!!!!!!
    # vorrei avere il tempo in input in termini di giorni o mesi ma non trovo come convertire 

    # Perform the averaging
    for i in range(new_length):
        start = i * window_size
        end = start + window_size
        averaged_data[:, i] = data[:, start:end].mean(axis=1)

    return averaged_data

np.random.seed(0)
n_time_points = 100
t = np.linspace(0, 4 * np.pi, n_time_points)
data = np.sin(t) + 0.2 * np.random.randn(1, n_time_points)  # Sine wave with noise

# Apply averaging over a window size of 10
averaged_data = average_time_series(data, 10, True)

# Plot the original data and the averaged data superimposed
plt.figure(figsize=(10, 6))
plt.plot(t, data[0, :], label='Original Data (Noisy)', color='blue')
# Scale the averaged data to match the time axis
t_averaged = np.linspace(0, 4 * np.pi, averaged_data.shape[1])
plt.plot(t_averaged, averaged_data[0, :], label='Averaged Data', color='red', linewidth=2)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Original and Averaged Time Series')
plt.show()


