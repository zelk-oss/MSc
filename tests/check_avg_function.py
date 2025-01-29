import numpy as np

# Function to convert days to data points
def convert_days_to_points(days):
    points_per_day = 1  # Assuming 11 points per day
    return int(days * points_per_day)

# Averaging function
def average_time_series(data, window_size_days, apply_averaging):
    if not apply_averaging:
        return data  # Return original data if averaging is not applied

    ntime, nvar = data.shape  # Correct shape extraction
    window_size = convert_days_to_points(window_size_days)

    if window_size > ntime:
        raise ValueError("Window size is larger than the time series length.")

    # Calculate the number of averages that can be computed
    new_length = ntime - window_size + 1
    print("new time series length: ", new_length)

    averaged_data = np.empty((new_length, nvar))

    # Perform moving average
    for i in range(new_length):
        start = i
        end = start + window_size
        averaged_data[i, :] = data[start:end, :].mean(axis=0)

    return averaged_data

# Example to test the function
np.random.seed(42)  # For reproducible random data
ntime = 50
nvar = 1

# Generate random data: rows are time points, columns are variables
data = np.random.random((ntime, nvar))
print(data)

# Perform moving average
window_size_days = 3
result = average_time_series(data, window_size_days, apply_averaging=True)

# Print the results
print("Original Data Shape:", data.shape)
print("\nMoving Averaged Data Shape:", result.shape)
print("\nMoving Averaged Data:")
print(result)
