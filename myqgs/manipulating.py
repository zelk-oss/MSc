"""
an exercise code to get acquainted with time series of real fields coming from maooam 
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Load the data
psi_a = np.load("psi_a.npy")
psi_o = np.load("psi_o.npy")
#print("shape of single arrays = ", psi_a.shape, psi_o.shape)

# Reshape them into (time, x*y)
a = psi_a.reshape(psi_a.shape[0], -1)
o = psi_o.reshape(psi_o.shape[0], -1)

# Concatenate along the second axis
concatenated = np.concatenate((a, o), axis=1) 
#print("shape of concatenated 2D object = ", concatenated.shape)  # time * length of concatenated thing

# Flatten into a 1D array
X = concatenated.reshape(-1)  # The full 1D vector
if X.shape[0] == psi_a.shape[0] * (psi_a.shape[1]*psi_a.shape[2] + psi_o.shape[1]*psi_o.shape[2]): 
    pass
else:
    print("Attention! Arrays sizes don't match")

# 1D data at a specified point in time
time_point = 300
X_t = concatenated[time_point, :]
cov = (1 / (X_t.shape[0] - 1)) * np.outer(X_t.T, X_t)

# Visualization of both psi
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size
norm = TwoSlopeNorm(vmin=cov.min(), vcenter=0, vmax=cov.max())
cax = ax.matshow(cov, cmap="RdBu", norm = norm)
fig.colorbar(cax)
ax.set_title(f"Covariance Matrix for Ψa and Ψo (in sequence) at time t = {time_point}", pad=20)
ax.set_xlabel("1D grid points")
ax.set_ylabel("1D grid points")


# Only streamfunction psi_a at a fixed point in time
psi_a_t = concatenated[time_point, 0 : a.shape[1]]  # (x*y, )
#print("shapes of a, 0 and 1 = ", a.shape[0], a.shape[1])
cov_psi_a = (1 / (psi_a_t.shape[0] - 1)) * np.outer(psi_a_t.T, psi_a_t)
#print(cov_psi_a.shape)

fig_psi_a, ax_psi_a = plt.subplots(figsize=(8, 6))
norm = TwoSlopeNorm(vmin=cov_psi_a.min(), vcenter=0, vmax=cov_psi_a.max())
cax_psi_a = ax_psi_a.matshow(
    cov_psi_a, cmap="RdBu", norm = norm
)
fig_psi_a.colorbar(cax_psi_a)
ax_psi_a.set_title(f"Covariance Matrix for psi_a at time t = {time_point}. 3x4 grid.", pad=20)
ax_psi_a.set_xlabel("grid points in 1D")
ax_psi_a.set_ylabel("grid points in 1D")

plt.show()


# Only streamfunction psi_o at a fixed point in time
psi_o_t = concatenated[time_point, a.shape[1] : ]  # (x*y, )
cov_psi_o = (1 / (psi_o_t.shape[0] - 1)) * np.outer(psi_o_t.T, psi_o_t)
print("shape of covariance of psi_o = ", cov_psi_o.shape)

fig_psi_o, ax_psi_o = plt.subplots(figsize=(8, 6))
cax_psi_o = ax_psi_o.matshow(
    cov_psi_o, cmap="RdBu")
fig_psi_o.colorbar(cax_psi_o)
ax_psi_o.set_title("Covariance Matrix for psi_o", pad=20)
ax_psi_o.set_xlabel("Variables")
ax_psi_o.set_ylabel("Variables")

#plt.show()

"""
now of course the plots are  not straightforward to interpret. The first line is the computed 
covariance of all there is in psi_a_t against the first element of it. The first element is the 
element (1,1) of the grid at time 0, in this case. The whole line is the whole grid. 
So the first line is the covariance of all grid points againt the first one. And so on. 
"""

"""
What are we interested in? Probably correlation between different functions. And at different
points of time too. The picture where we look at correlation between different functions shows 
us correlation between the second function and itself, and zero correlation in all other places. 
I would like to see more in detail the correlation of psi_a against psi_o. """

cov_a_o = cov[a.shape[1]:, :a.shape[1]]
print("covariance of a against o = ", cov_a_o)
# correlation for psi_a against psi_o (and vice versa, it's symmetric)
fig2, ax2 = plt.subplots(figsize=(8, 6))  # Adjust figure size
cax2 = ax2.matshow(cov_a_o, cmap="RdBu")
fig2.colorbar(cax2)
ax2.set_title("Second Quad of Covariance Matrix", pad=20)
ax2.set_xlabel("Variables")
ax2.set_ylabel("Variables")

print("psi_a_t - psi_o_t = ", psi_a_t - psi_o_t)

#plt.show()