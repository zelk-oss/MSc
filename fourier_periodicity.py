import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure plot styles
mpl.rcParams.update({
    'font.size': 26,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'legend.fontsize': 14,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22
})

# Function to plot a single component
def plot_single(file_path, which_component, zoom=1, stride=1):
    """
    Plot a single component with zoom and stride options.

    Parameters:
        file_path (str): Path to the data file.
        which_component (int): Index of the component to plot.
        zoom (int): Fraction of the time series to visualize (e.g., 10 = 1/10).
        stride (int): Show every N-th data point (e.g., 10 = every 10th point).
    """
    data = np.loadtxt(file_path)
    print(f"Loaded file '{file_path}', shape: {data.shape}")

    time_points = data[:, 0]
    max_time = time_points[-1] * 0.11 / 365.24  # convert to years
    time = np.linspace(0, max_time, len(time_points))
    component = data[:, which_component]

    # Apply zoom and stride
    limit = len(time) // zoom
    time = time[:limit:stride]
    component = component[:limit:stride]

    plt.figure(figsize=(15, 9))
    plt.plot(time, component, label=f"component {which_component}", color='darkblue', marker ="o", markersize=2)
    plt.title(f"component {which_component}")
    plt.xlabel("time (years)")
    plt.ylabel(f"component {which_component}")
    plt.legend()
    plt.grid()
    plt.show()

# Function to superimpose two components from different files
def plot_superimposed(series_list, zoom=1, stride=1):
    """
    Superimpose an arbitrary number of components from one or more files and save the plot.

    Parameters:
        series_list (list of tuples): Each tuple is (file_path, component_index).
        zoom (int): Fraction of time series to visualize.
        stride (int): Show every N-th data point.
    """
    plt.figure(figsize=(15, 9))

    filenames = []

    for i, (file_path, comp_idx) in enumerate(series_list):
        data = np.loadtxt(file_path)
        time_pts = data[:, 0]
        time = np.linspace(0, time_pts[-1] * 0.11 / 365.24, len(time_pts))
        component = data[:, comp_idx]

        # Apply zoom and stride
        limit = len(time) // zoom
        time = time[:limit:stride]
        component = component[:limit:stride]

        # Determine coupling type
        if "1_1e-7" in file_path:
            coupling = "strong"
        elif "1e-8" in file_path:
            coupling = "weak"
        else:
            coupling = "unknown"

        # Track file description for filename
        filenames.append(f"{comp_idx}_{coupling}")

        plt.plot(time, component, label=f"component {comp_idx} ({coupling} coupling)", 
                 marker="o", markersize=2)

    plt.title("superimposed components")
    plt.xlabel("time (years)")
    plt.ylabel("amplitude")
    plt.legend()
    plt.grid()
    plt.show()

    # Generate output filename
    filename_base = "_".join(filenames)
    output_filename = f"{filename_base}.png"
    #plt.savefig(output_filename, dpi=300)
    #plt.close()
    #print(f"Plot saved as '{output_filename}'")



# MAIN
strong_coupling = "../data_thesis/data_1e5points_1000ws/evol_fields_1_1e-7.dat"
weak_coupling = "../data_thesis/data_1e5points_1000ws/evol_fields_1e-8.dat"

# Examples:
#plot_single(weak_coupling, which_component=1, zoom=10, stride=5)
#plot_single(strong_coupling, which_component=1, zoom=10, stride=5)


"""
# first 10 atmospheric components 
plot_superimposed([
    (strong_coupling, 1),
    #(strong_coupling, 2),
    #(strong_coupling, 3), 
    #(strong_coupling, 4),
    #(strong_coupling, 5),
    (strong_coupling, 6),
    #(strong_coupling, 7),
    #(strong_coupling, 8),
    (strong_coupling, 9),
    #(strong_coupling, 10)
], zoom=10, stride=1)


# second 10 atmospheric c 
plot_superimposed([
    (strong_coupling, 11),
    #(strong_coupling, 12),
    #(strong_coupling, 13), 
    #(strong_coupling, 14), 
    #(strong_coupling, 15), 
    (strong_coupling, 16), 
    #(strong_coupling, 17),
    #(strong_coupling, 18), 
    (strong_coupling, 19), 
    #(strong_coupling, 20)        
], zoom=10, stride=1)
"""

"""
# first 8 ocean c 
plot_superimposed([
    (strong_coupling, 21),
    (strong_coupling, 22),
    #(strong_coupling, 23), 
    #(strong_coupling, 24), 
    #(strong_coupling, 25), 
    (strong_coupling, 26), 
    #(strong_coupling, 27),
    #(strong_coupling, 28)        
], zoom=10, stride=1)



# second 8 ocean c 
plot_superimposed([
    (strong_coupling, 29),
    (strong_coupling, 30),
    #(strong_coupling, 31),
    (strong_coupling, 32), 
    #(strong_coupling, 33),
    (strong_coupling, 34),
    #(strong_coupling, 35),
    #(strong_coupling, 36)
], zoom = 10, stride=1)

#######################
# weak coupling section 
#######################

# first 10 atmospheric components 
plot_superimposed([
    (weak_coupling, 1),
    #(strong_coupling, 2),
    #(strong_coupling, 3), 
    #(strong_coupling, 4),
    #(strong_coupling, 5),
    (weak_coupling, 6),
    #(strong_coupling, 7),
    #(strong_coupling, 8),
    (weak_coupling, 9),
    #(strong_coupling, 10)
], zoom=10, stride=1)


# second 10 atmospheric c 
plot_superimposed([
    (weak_coupling, 11),
    #(strong_coupling, 12),
    #(strong_coupling, 13), 
    #(strong_coupling, 14), 
    #(strong_coupling, 15), 
    (weak_coupling, 16), 
    #(strong_coupling, 17),
    #(strong_coupling, 18), 
    (weak_coupling, 19), 
    #(strong_coupling, 20)        
], zoom=1, stride=1)

# first 8 ocean c 
plot_superimposed([
    (weak_coupling, 21),
    (weak_coupling, 22),
    #(strong_coupling, 23), 
    #(strong_coupling, 24), 
    #(strong_coupling, 25), 
    (weak_coupling, 26), 
    #(strong_coupling, 27),
    #(strong_coupling, 28)        
], zoom=1, stride=1)

# second 8 ocean c 
plot_superimposed([
    (weak_coupling, 29),
    (weak_coupling, 30),
    #(strong_coupling, 31),
    (weak_coupling, 32), 
    #(strong_coupling, 33),
    (weak_coupling, 34),
    #(strong_coupling, 35),
    #(strong_coupling, 36)
], zoom = 1, stride=1)

"""

# plotting ocean and atmosphere components on the same plot now 
plot_superimposed([
    (weak_coupling, 1),
    (weak_coupling, 6),
    (weak_coupling, 22),
    (weak_coupling, 30), 
    (weak_coupling, 34)
], zoom = 10, stride=1)

plot_superimposed([
    (strong_coupling, 1),
    (strong_coupling, 6),
    (strong_coupling, 22),
    (strong_coupling, 30), 
    (strong_coupling, 34)
], zoom = 10, stride=1)


"""
# now superimpose strong and weak coupling example components 
plot_superimposed([
    (strong_coupling, 1),
    (weak_coupling, 1)
], zoom = 1, stride=1)

plot_superimposed([
    (strong_coupling, 22),
    (weak_coupling, 22)
], zoom = 1, stride=1)
"""