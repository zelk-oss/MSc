import numpy as np 
import matplotlib.pyplot as plt
import csv

# importing Liang's bivariate formula from another folder 
import sys 
sys.path.insert(0, '/home/chiaraz/Liang_Index_climdyn')
#from function_liang import compute_liang
from function_liang_nvar import compute_liang_nvar
import time 

start = time.time()

# add annotations 
def add_annotations(ax, data):
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=7)

# Initialize an empty 36x36 matrix to store the results of compute_liang
result_matrix = np.empty((36, 36), dtype=object)
data_in_file = []

# opening the time series file in /qgs directory  
file_path = '../qgs/evol_fields_3e-7.dat'
with open(file_path, 'r') as file:
    for index, line in enumerate(file):
        if index % 1 == 0:
            # handling potential invalid data 
            try:
                # turns lines in the file in tuples of 37 elements in the maooam case
                row = [float(value) for value in line.split()]
                if len(row) < 37: 
                    print("Line has insufficient data: {line.strip()}")
                    continue 
                # here we can do stuff with the data 
                data_in_file.append(row)

            except ValueError:
                print(f"Could not convert line to floats: {line}")

# Convert the data list to a numpy 2D array (shape: N x 37)
time_series = np.array(data_in_file) 
n_columns = len(time_series[0]) # should be 37 

#import time
#start = time.time()
#result = compute_liang(time_series[:, 1], time_series[:, 2], 1, 20)
#end = time.time()
#print("Time for one computation (bivariate):", end - start)


"""
# Iterate over all pairs of columns (i, j)
for i in range(1, n_columns):
    for j in range(1, n_columns):
        # Extract the i-th and j-th columns from the numpy array (time_serie)
        column_i = time_series[:, i]  # All rows, i-th column
        column_j = time_series[:, j]  # All rows, j-th column
        # Apply the compute_liang function and store the result in result_matrix
        # elements in compute_liang result: 
        # 0: T21
        # 1: tau21
        # 2: error_T21
        # 3: error_tau21
        # 4: R
        # 5: error R 
        # 6: error_T21_FI
        result_matrix[i-1, j-1] = compute_liang(column_i, column_j, 10, 20)
# Now result_matrix is a 36x36 matrix, where each element is the output of compute_liang for the corresponding column pair

tau_matrix = np.array([[np.abs(result_matrix[i, j][1]) for j in range(36)] for i in range(36)])
r_matrix = np.array([[result_matrix[i, j][4] for j in range(36)] for i in range(36)])

# bivariate plot, all variables 
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the tau21 matrix
cax1 = axs[0].imshow(tau_matrix, aspect='auto', cmap='Greens')
axs[0].set_title('Tau21 Matrix')
axs[0].set_xlabel('Column Index')
axs[0].set_ylabel('Row Index')
fig.colorbar(cax1, ax=axs[0])

# Plot the r_matrix (element 4 values)
cax2 = axs[1].imshow(r_matrix, aspect='auto', cmap='RdBu')
axs[1].set_title('Element 4 Matrix (R)')
axs[1].set_xlabel('Column Index')
axs[1].set_ylabel('Row Index')
fig.colorbar(cax2, ax=axs[1])

# Salvataggio in CSV
output_data = [[f"[{i},{j}]", tau_matrix[i, j], r_matrix[i, j]] for i in range(36) for j in range(36)]
with open("tau_r_matrix.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Index", "Tau Value", "R Value"])
    writer.writerows(output_data)

"""
# allora dato che questa versione monovariate calcola la covarianza fra le singole variabili, 
# è chiaro che darà risultati un po' così sulla diagonale. perché la covarianza di x1 versus 
# x1 è zero e se poi va al denominatore siamo spacciati. 
# la mia ipotesi è che possiamo avere cose diverse, che sono poi quelle che ottiene docquier, 
# facendo la multivariate 

###################################################
# Analysis considering only first vars
###################################################

# Analisi su variabili principali
columns_to_extract = [1, 11, 21, 29]
principal_series = time_series[1000:1200, columns_to_extract]

#result_matrix_4x4 = np.array([[compute_liang(principal_series[:, i], principal_series[:, j], 1, 20)
#                               for j in range(4)] for i in range(4)])
#tau_4x4 = np.abs([[result_matrix_4x4[i][j][1] for j in range(4)] for i in range(4)])
#r_4x4 = [[result_matrix_4x4[i][j][4] for j in range(4)] for i in range(4)]


# multivariate analysis for selected columns 
nvar_results = np.array(compute_liang_nvar(principal_series, 1, 20)) 
tau_nvar = np.empty((4,4))
r_nvar = np.empty((4,4))
for i in range(4): 
    for j in range(4): 
        tau_nvar[i,j] = np.abs(nvar_results[1,i,j])
        r_nvar[i,j] = nvar_results[4,i,j]
        print("element (", i ,",", j ,") added")

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

"""
# Tau plot
cax1 = axs[0].imshow(tau_4x4, aspect='auto', cmap='Greens')
axs[0].set_title('Tau21 Matrix (4 variables)')
axs[0].set_xlabel('Variable Index')
axs[0].set_ylabel('Variable Index')
fig.colorbar(cax1, ax=axs[0])
add_annotations(axs[0], tau_4x4)

# R plot
cax2 = axs[1].imshow(r_4x4, aspect='auto', cmap='RdBu')
axs[1].set_title('R Matrix (4 variables)')
axs[1].set_xlabel('Variable Index')
axs[1].set_ylabel('Variable Index')
fig.colorbar(cax2, ax=axs[1])
add_annotations(axs[1], r_4x4)
"""


fig_nvar, axs_nvar = plt.subplots(1, 2, figsize=(12, 6))

# Tau nvar plot 
cax1_nvar = axs_nvar[0].imshow(tau_nvar, aspect='auto', cmap='Greens')
axs_nvar[0].set_title('Tau21 Matrix (4 variables)')
axs_nvar[0].set_xlabel('Variable Index')
axs_nvar[0].set_ylabel('Variable Index')
fig_nvar.colorbar(cax1_nvar, ax=axs_nvar[0])
add_annotations(axs_nvar[0], tau_nvar)

print("i got here")
# R nvar plot
cax2_nvar = axs_nvar[1].imshow(r_nvar, aspect='auto', cmap='RdBu')
axs_nvar[1].set_title('R Matrix (4 variables)')
axs_nvar[1].set_xlabel('Variable Index')
axs_nvar[1].set_ylabel('Variable Index')
fig_nvar.colorbar(cax2_nvar, ax=axs_nvar[1])
add_annotations(axs_nvar[1], r_nvar)

###################################################
# attractor anaylsis 
###################################################

"""
tau_attr = np.empty((3,3))
r_attr = np.empty((3,3))

# multivariate analysis
attr_series = time_series[:,[1,22,30]]
attr_results = np.array(compute_liang_nvar(attr_series, 10, 20)) 

for i in range(3):
    for j in range(3):
        tau_attr[i,j] = np.abs(attr_results[1,i,j])
        r_attr[i,j] = attr_results[4,i,j]

# Create the plots for tau21 and R matrices for the 4 selected variables
fig_attr, axs_attr = plt.subplots(1, 2, figsize=(12, 6))

# Plot the tau21 matrix for the 3 selected variables
cax1_attr = axs_attr[0].imshow(tau_attr, aspect='auto', cmap='Greens')
axs_attr[0].set_title('Tau21 Matrix (3 variables)')
axs_attr[0].set_xlabel('Variable Index')
axs_attr[0].set_ylabel('Variable Index')
fig_attr.colorbar(cax1_attr, ax=axs[0])
add_annotations(axs_attr[0], axs_attr)
# Plot the R matrix for the 3 lected variables
cax2_attr = axs_attr[1].imshow(r_attr, aspect='auto', cmap='RdBu')
axs_attr[1].set_title('R Matrix (3 variables)')
axs_attr[1].set_xlabel('Variable Index')
axs_attr[1].set_ylabel('Variable Index')
fig_attr.colorbar(cax2_attr, ax=axs_attr[1])
add_annotations(axs_attr[1], axs_attr)
"""

end = time.time() 
print(end - start)
plt.tight_layout()
plt.show()