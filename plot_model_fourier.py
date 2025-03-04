"""
Here I plot some useful things about the model. 
The attractor, comparison of attractors for different couplings,
and even the coefficients multiplying the Fourier components of the model.
"""

import matplotlib.pyplot as plt
import numpy as np 

# a variable to make plotting faster (less points)
accelerate = 1

# file a 
data_in_file_a = []
# opening the time series file in /qgs directory  
file_path = '../data_thesis/data_1e5points_1000ws/evol_fields_1e-8.dat'
with open(file_path, 'r') as file:
    for index, line in enumerate(file):
        if index % accelerate == 0:
            # handling potential invalid data 
            try:
                # turns lines in the file in tuples of 37 elements in the maooam case
                row = [float(value) for value in line.split()]
                if len(row) < 37: 
                    print("Line has insufficient data: {line.strip()}")
                    continue 
    
                data_in_file_a.append(row)
                # at the end of the cycle all the data from the file in this becautiful array  

            except ValueError:
                print(f"Could not convert line to floats: {line}")
data_in_file_a = np.array(data_in_file_a)

# check what qgs is doing in terms of length of the series 
#print(len(data_in_file_a))

# file b  
data_in_file_b = [] 
file_path = '../data_thesis/data_1e5points_1000ws/evol_fields_1_1e-7.dat'
with open(file_path, 'r') as file:
    for index, line in enumerate(file):
        if index % accelerate == 0:
            # handling potential invalid data 
            try:
                # turns lines in the file in tuples of 37 elements in the maooam case
                row = [float(value) for value in line.split()]
                if len(row) < 37: 
                    print("Line has insufficient data: {line.strip()}")
                    continue 
    
                data_in_file_b.append(row)
                # at the end of the cycle all the data from the file in this becautiful array  

            except ValueError:
                print(f"Could not convert line to floats: {line}")
data_in_file_b = np.array(data_in_file_b)

"""
# file c   
data_in_file_c = []
file_path = 'myqgs/evol_fields_1e-8.dat'
with open(file_path, 'r') as file:
    for index, line in enumerate(file):
        if index % accelerate == 0:
            # handling potential invalid data 
            try:
                # turns lines in the file in tuples of 37 elements in the maooam case
                row = [float(value) for value in line.split()]
                if len(row) < 37: 
                    print("Line has insufficient data: {line.strip()}")
                    continue 
    
                data_in_file_c.append(row)
                # at the end of the cycle all the data from the file in this becautiful array  

            except ValueError:
                print(f"Could not convert line to floats: {line}")
data_in_file_c = np.array(data_in_file_c)
"""

"""
# opening second time series file in /qgs directory 
data_in_file_d = [] 
file_path = 'myqgs/evol_fields_0_8e-7.dat'
with open(file_path, 'r') as file:
    for index, line in enumerate(file):
        if index % accelerate == 0:
            # handling potential invalid data 
            try:
                # turns lines in the file in tuples of 37 elements in the maooam case
                row = [float(value) for value in line.split()]
                if len(row) < 37: 
                    print("Line has insufficient data: {line.strip()}")
                    continue 
    
                data_in_file_d.append(row)
                # at the end of the cycle all the data from the file in this becautiful array  

            except ValueError:
                print(f"Could not convert line to floats: {line}")
data_in_file_d = np.array(data_in_file_d)
"""

def other_stuff():
    # 3d plot for the attractor for different values of d 
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_title("Comparison of attractors in 3d")
    ax.scatter(data_in_file_a[:,22], data_in_file_a[:,30], data_in_file_a[:,1], s=0.1, label = r"$d = 1e^{-8}$")
    # plot attractor with time series of second file 
    ax.scatter(data_in_file_b[:,22], data_in_file_b[:,30], data_in_file_b[:,1], s=0.1, label=r"$d = 1.1e^{-7}$")
    #ax.scatter(data_in_file_c[:,22], data_in_file_c[:,30], data_in_file_c[:,1], s=0.1, label =r"$d = 1e^{-8}$")
    #ax.scatter(data_in_file_d[:,22], data_in_file_d[:,30], data_in_file_d[:,1], s=0.1, label = r"$d = 0.8e^{-7}$")
    ax.set_xlabel(r"$\Psi_{o,2}$")
    ax.set_ylabel(r"$\delta T_{o,2}$")
    ax.set_zlabel(r"$\Psi_{a,1}$")
    ax.legend(loc="best", markerscale = 20)


    # only the coupled one (where the attractor is cute)
    fig8 = plt.figure(figsize=(10, 8))
    ax8 = fig8.add_subplot(111, projection = '3d')
    ax8.set_title("Attractor in 3d")
    ax8.scatter(data_in_file_b[:,22], data_in_file_b[:,30], data_in_file_b[:,11], s=0.1, label=r'$d = 1.1e^{-7}$')
    ax8.set_xlabel(r"$\Psi_{o,2}$")
    ax8.set_ylabel(r"$\delta T_{o,4}$")
    ax8.set_zlabel(r"$\Psi_{a,1}$")
    ax8.legend(loc="best", markerscale = 20)

    #plt.show()


    # plotting projections and fourier components 
    data_in_file = []
    file_path = '../data_thesis/data_1e5points_1000ws/evol_fields_1_1e-7.dat'
    with open(file_path, 'r') as file:
        for index, line in enumerate(file):
            if index % accelerate == 0: 
                # turns lines in the file in tuples of 37 elements in the maooam case
                row = [float(value) for value in line.split()]    
                data_in_file.append(row)

    data_in_file = np.array(data_in_file)
    print(np.shape(data_in_file))


    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111)
    ax1.set_title("2D projection of the attractor, strong coupling")
    ax1.scatter(data_in_file[:,22], data_in_file[:,30], s=0.1)
    ax1.set_xlabel(r"$\Psi_{o,2}$")
    ax1.set_ylabel(r"$\delta T_{o,2}$")
    ax1.legend(loc="best", markerscale = 20)

    #plt.show()


    # 2d plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid of subplots
    fig.suptitle("Atmosphere and Ocean Data")  # Add a common title to the figure

    # Atmosphere 1-10
    axs[0, 0].set_title("Atmosphere 1-10: barotropic streamfunction")
    for i in range(1, 11):
        axs[0, 0].scatter(data_in_file[:, 0], data_in_file[:, i], marker="o", s=1, label=f'function {i}')
    axs[0,0].legend(loc="upper left", markerscale = 10)

    # Atmosphere 11-20
    axs[0, 1].set_title("Atmosphere 11-20: baroclinic streamfunction")
    for j in range(11, 21):
        axs[0, 1].scatter(data_in_file[:, 0], data_in_file[:, j], marker="o", s=1, label=f'function {j}')
    axs[0,1].legend(loc="upper left", markerscale = 10)

    print("this is the number of points = ", len(data_in_file[:,0]))
    # Ocean 20-28
    axs[1, 0].set_title("Ocean 21-28: oceanic streamfunction")
    for m in range(21, 29):
        axs[1, 0].scatter(data_in_file[:, 0], data_in_file[:, m], marker="o", s=1, label=f'function {m}')
    axs[1,0].legend(loc="upper left", markerscale = 10)

    # Ocean 29-36
    axs[1, 1].set_title("Ocean 29-36: oceanic temperature")
    for k in range(29, 37):
        axs[1, 1].scatter(data_in_file[:, 0], data_in_file[:, k], marker="o", s=1, label=f'function {k}')
    axs[1,1].legend(loc="best", markerscale = 10)
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout and leave space for the common title


#plt.show()

def plot_attractors_side_by_side(weak_data, strong_data, x_idx=22, y_idx=30, z_idx=1):
    """
    Plots the attractors from two datasets side by side in 3D.
    
    Parameters:
      strong_data (np.array): Array of data for strong coupling.
      weak_data (np.array): Array of data for weak coupling.
      x_idx (int): Column index for the x-axis variable.
      y_idx (int): Column index for the y-axis variable.
      z_idx (int): Column index for the z-axis variable.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Update rcParams to increase font sizes
    mpl.rcParams.update({
        'font.size': 26,         # Default text size
        'axes.titlesize': 24,    # Title size for axes
        'axes.labelsize': 24,    # Axis label size
        'legend.fontsize': 26,   # Legend text size
        'xtick.labelsize': 22,   # X-tick label size
        'ytick.labelsize': 22    # Y-tick label size
    })
    
    fig = plt.figure(figsize=(26, 12))
    
    # Left subplot for weak coupling attractor
    ax2 = fig.add_subplot(121, projection='3d')
    ax2.scatter(weak_data[:, x_idx], weak_data[:, y_idx], weak_data[:, z_idx],
                s=0.1, color='darkblue', label=r"attractor for $d = 1e^{-8}$")
    ax2.set_title("weak coupling", fontsize = 30)
    ax2.set_xlabel(r"$\Psi_{o,2}$", labelpad=35)
    ax2.set_ylabel(r"$\delta T_{o,2}$", labelpad=35)
    ax2.set_zlabel(r"$\Psi_{a,1}$", labelpad=35)
    ax2.legend(loc="best", markerscale=14)  # Increase legend marker size
    ax2.tick_params(axis='x', pad=14)
    ax2.tick_params(axis='y', pad=14)
    ax2.tick_params(axis='z', pad=14)
    
    # Right subplot for strong coupling attractor
    ax1 = fig.add_subplot(122, projection='3d')
    ax1.scatter(strong_data[:, x_idx], strong_data[:, y_idx], strong_data[:, z_idx],
                s=0.1, color='darkblue', label=r"attractor for $d = 1.1e^{-7}$")
    ax1.set_title("strong coupling", fontsize = 30)
    ax1.set_xlabel(r"$\Psi_{o,2}$", labelpad=35)
    ax1.set_ylabel(r"$\delta T_{o,2}$", labelpad=35)
    ax1.set_zlabel(r"$\Psi_{a,1}$", labelpad=35)
    ax1.legend(loc="best", markerscale=14)  # Increase legend marker size
    ax1.tick_params(axis='x', pad=14)
    ax1.tick_params(axis='y', pad=14)
    ax1.tick_params(axis='z', pad=14)
    
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming data_in_file_b contains strong coupling data 
# and data_in_file_a contains weak coupling data:
plot_attractors_side_by_side(data_in_file_a, data_in_file_b)
