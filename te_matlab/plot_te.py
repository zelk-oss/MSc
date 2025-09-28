import numpy as np
import matplotlib.pyplot as plt

"""
Plotting results from gaussian copula mutual information. 
O = X1
A = X2 
I should find that X2 causes X1 and not the vice versa 
"""

# Load saved results
data = np.load("te_results_mu2.npz")
TE_O_to_A = data["TE_O_to_A"] # (75 * 75), TE swept over all combinations of m and L 
TE_A_to_O = data["TE_A_to_O"] # (75 * 75) 
m_tot = data["m_tot"] # (75), history length 
L_tot = data["L_tot"] # (75), lag values 


# Index arrays
m_indices = np.arange(len(m_tot))
L_indices = np.arange(len(L_tot))
m_tick_pos = np.arange(0, len(m_indices), 10)
L_tick_pos = np.arange(0, len(L_indices), 10)

# plot 

# find common max btw the two plots 
vmin = min(TE_O_to_A.min(), TE_A_to_O.min())
vmax = max(TE_O_to_A.max(), TE_A_to_O.max())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im1 = axes[0].pcolormesh(m_indices, L_indices, TE_O_to_A.T,
                         shading="auto", vmin=vmin, vmax=vmax)
axes[0].set_xticks(m_tick_pos)
axes[0].set_xticklabels(m_tot[m_tick_pos])
axes[0].set_yticks(L_tick_pos)
axes[0].set_yticklabels(L_tot[L_tick_pos])
axes[0].set_title("X1 → X2")
axes[0].set_xlabel("history length")
axes[0].set_ylabel("lag")
fig.colorbar(im1, ax=axes[0], label='TE')

im2 = axes[1].pcolormesh(m_indices, L_indices, TE_A_to_O.T,
                         shading="auto", vmin=vmin, vmax=vmax)
axes[1].set_xticks(m_tick_pos)
axes[1].set_xticklabels(m_tot[m_tick_pos])
axes[1].set_yticks(L_tick_pos)
axes[1].set_yticklabels(L_tot[L_tick_pos])
axes[1].set_title("X2 → X1")
axes[1].set_xlabel("history length")
axes[1].set_ylabel("lag")
fig.colorbar(im2, ax=axes[1], label='TE')

plt.show()