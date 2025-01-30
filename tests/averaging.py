import numpy as np 
import matplotlib.pyplot as plt 

import sys 
sys.path.insert(0, '/home/chiaraz/thesis')

from functions_for_maooam import average_time_series

def generate_noisy_sine_array(num_signals=37, length=10000, noise_level=0.1):
    """
    Generate an array of shape (num_signals, length) filled with noisy sine waves.
    """
    time = np.linspace(0, 10 * np.pi, length)
    data = np.array([
        np.sin(time + np.random.uniform(0, 2*np.pi)) + noise_level * np.random.randn(length)
        for _ in range(num_signals)
    ])
    return data

def visualize_line(data, line_index=0):
    """
    Visualize a specific line of the dataset.
    """
    if line_index < 0 or line_index >= data.shape[0]:
        raise ValueError("Invalid line index.")
    
    plt.figure(figsize=(10, 4))
    plt.plot(data[line_index], label=f"Line {line_index}")
    plt.xlabel("Time step")
    plt.ylabel("Amplitude")
    plt.title(f"Visualization of Line {line_index}")
    plt.legend()


data = generate_noisy_sine_array(37, 1000, 0.1)

# Apply averaging over a window size of 10
averaged_data = average_time_series(data, 110, True)

visualize_line(averaged_data, 0)
visualize_line(averaged_data, 1) 
visualize_line(averaged_data, 20) 
visualize_line(averaged_data, 21)

plt.show()

