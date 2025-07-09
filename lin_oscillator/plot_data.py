import numpy as np
import matplotlib.pyplot as plt

# Load the data, skipping the header row
data = np.loadtxt("data_mu0_n138949.txt", skiprows=1)

# Extract columns
time = data[:, 0]
ocean = data[:, 1]
atmosphere = data[:, 2]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(time, ocean, label="Ocean", color='blue')
plt.plot(time, atmosphere, label="Atmosphere", color='orange')
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Ocean and Atmosphere vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
