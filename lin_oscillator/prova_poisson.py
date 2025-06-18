import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# Parameters
n_samples = 100000  # Number of i.i.d. samples
distribution = np.random.uniform  # You can change to np.random.normal, etc.

# Step 1: Generate i.i.d. variables (Uniform on [0,1])
samples = distribution(size=n_samples)

# Step 2: Sort the values
sorted_samples = np.sort(samples)

# Step 3: Compute spacings (differences between successive sorted values)
spacings = np.diff(sorted_samples)

# Step 4: Normalize spacings (mean should be ~1)
spacings *= n_samples  # Because uniform spacing ~ 1/n, so we scale

# Step 5: Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(spacings, bins=100, density=True, alpha=0.6, color='skyblue', label='Empirical spacings')

# Step 6: Overlay exponential PDF
x = np.linspace(0, 5, 200)
plt.plot(x, expon.pdf(x), 'r-', lw=2, label='Exponential PDF (λ = 1)')

# Labels and legend
plt.title("Spacing Distribution Between Sorted i.i.d. Variables (Uniform)")
plt.xlabel("Spacing")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
