"""
Analyze saved time series using Liang index.
"""

import numpy as np
import sys
from jpype import *
import os 

# === Load time series ===
input_file = "/home/chiaraz/thesis/lin_oscillator/2D_system_data/2D_timeseries.txt"
data = np.loadtxt(input_file, skiprows=1)
t = data[:, 0]
X1 = data[:, 1]
X2 = data[:, 2]

# === Liang index analysis ===
sys.path.append('/home/chiaraz/Liang_Index_climdyn/')
from function_liang_nvar import compute_liang_nvar

dt = 0.01
# bootstrap iterations 
n_iter = 200
conf = 1.96
nvar = 2
start = int(10 / dt)

xx = np.array((X1[start:], X2[start:]))
T, tau, R, error_T, error_tau, error_R = compute_liang_nvar(xx, dt, n_iter)

def compute_sig(var, error, conf):
    if (var - conf * error < 0. and var + conf * error < 0.) or (var - conf * error > 0. and var + conf * error > 0.):
        return 1
    else:
        return 0

# Compute significance
sig_T = np.zeros((nvar, nvar))
sig_tau = np.zeros((nvar, nvar))
sig_R = np.zeros((nvar,nvar))
for j in range(nvar):
    for k in range(nvar):
        sig_T[j, k] = compute_sig(T[j, k], error_T[j, k], conf)
        sig_tau[j,k] = compute_sig(tau[j,k], error_tau[j,k], conf)
        sig_R[j,k] = compute_sig(R[j,k],error_R[j,k],conf)

print("=== Liang Index ===")
print("T matrix:\n", T)
print("Significance (T):\n", sig_T)
print("tau matrix:\n", tau)
print("Significance tau:\n", sig_tau)
print("R matrix:\n", R)
print("Significance R:\n", sig_R)

output_dir = os.path.expanduser("~/thesis/lin_oscillator/2D_system_data")
os.makedirs(output_dir, exist_ok=True)
filename = f"liang_2D_bias1.txt"
def write_matrix(file, name, matrix):
    file.write(f"{name}:\n")
    file.write(np.array2string(matrix, precision=8, suppress_small=True))
    file.write("\n\n")

with open(os.path.join(output_dir, filename), 'w') as file:
    file.write("=== Liang Index ===\n\n")
    write_matrix(file, "T matrix", T)
    write_matrix(file, "Significance (T)", sig_T)
    write_matrix(file, "tau matrix", tau)
    write_matrix(file, "Significance tau", sig_tau)
    write_matrix(file, "R matrix", R)
    write_matrix(file, "Significance R", sig_R)
print("saved file")