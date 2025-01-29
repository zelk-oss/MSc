import numpy as np
import matplotlib.pyplot as plt

# Function to analyze sinusoidal time series and frequency over time windows
def analyze_sinusoidal(file_path):
    """
    Analyze and plot sinusoidal time series for all columns in a data file, 
    and plot frequency as a function of increasing time window.
    
    Parameters:
        file_path (str): Path to the data file.
    """
    # Load the data file
    data = np.loadtxt(file_path)
    print(f"Data shape: {data.shape}")
    
    # First column as time
    time_points = data[:, 0]
    max_time = time_points[len(time_points)-1]*0.11 / (365.24/12)
    time = np.linspace(0, max_time, len(time_points))
    total_time = time[-1] - time[0]
    data_interval = np.mean(np.diff(time))  # Determine time interval from time column
    print("data interval", data_interval)
    
    # Loop through all columns
    for column_index in range(11,12):
    #range(20, data.shape[1]):
        # Extract the time series
        time_series = data[:, column_index]
        
        # Calculate the mean of the series for oscillation reference
        mean_value = np.mean(time_series)
        
        # Find crossings around the mean
        zero_crossings = np.where(np.diff(np.sign(time_series - mean_value)))[0]
        
        # Calculate periods (time between crossings)
        crossing_times = time[zero_crossings]
        half_periods = np.diff(crossing_times)
        periods = half_periods[::2] * 2  # Full periods, assuming symmetry
        
        # Calculate overall frequencies
        average_period = np.mean(periods) if len(periods) > 0 else np.inf
        frequency_per_day = 1 / average_period if average_period < np.inf else 0
        frequency_per_year = frequency_per_day * 365.25
        
        # Print overall frequencies
        print(f"Column {column_index}:")
        print(f"  Mean value: {mean_value:.6f}")
        print(f"  Frequency: {frequency_per_day:.6f} cycles per day")
        print(f"  Frequency: {frequency_per_year:.6f} cycles per year")
        
        # Plot the sinusoidal function
        sparse_time_series = time_series[0::1]
        sparse_time = time[0::1]
        plt.figure(figsize=(10, 6))
        plt.plot(time, time_series, label=r'$\delta T_{a,1}$ Time Series', linestyle='-', color='seagreen')
        plt.scatter(sparse_time, sparse_time_series, color='red', label='Data Points', s=4, zorder=5)
        #plt.axhline(mean_value, color='black', linestyle='--', linewidth=0.8, label='Mean Value')
        plt.title(r'$\delta T_{a,1}$ Time Series')
        plt.xlabel('Time (months)')
        plt.ylabel(r'$\delta T_{a,1}$')
        plt.legend()
        plt.grid()
        plt.show()
        """
        # Frequency as a function of time window
        frequencies = []
        windows = np.linspace(0, total_time, 100)  # Create 100 time windows
        for t_max in windows:
            indices = time <= t_max
            window_times = time[indices]
            window_series = time_series[indices]
            
            # Calculate zero-crossings and periods within the window
            window_mean = np.mean(window_series)
            window_zero_crossings = np.where(np.diff(np.sign(window_series - window_mean)))[0]
            window_crossing_times = window_times[window_zero_crossings]
            window_half_periods = np.diff(window_crossing_times)
            window_periods = window_half_periods[::2] * 2
            
            # Calculate frequency for the current window
            if len(window_periods) > 0:
                avg_period = np.mean(window_periods)
                frequencies.append(1 / avg_period if avg_period > 0 else 0)
            else:
                frequencies.append(0)
        
        # Plot frequency as a function of time window
        plt.figure(figsize=(10, 6))
        plt.plot(windows, frequencies, label=f'Frequency Evolution (Column {column_index})', color='green')
        plt.title(f'Frequency as a Function of Time Window (Column {column_index})')
        plt.xlabel('Time Window (days)')
        plt.ylabel('Frequency (cycles per day)')
        plt.legend()
        plt.grid()
        #plt.show()
        """

# Example usage
file_path = "../data_toobig/data_1e5points_1000ws/evol_fields_1e-8.dat"  # Replace with your file path
analyze_sinusoidal(file_path)
