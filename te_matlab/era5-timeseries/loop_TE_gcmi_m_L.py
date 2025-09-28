import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append(".")
from gcmi import *

# --- Step 1: parameter grids ---

# m and L values are log-distributed between 10^0 and 10^3 
m_tot = np.unique(np.round(np.logspace(0, 2.4, 100)).astype(int))
L_tot = np.unique(np.round(np.logspace(0, 2.4, 100)).astype(int))
m_indices = np.arange(len(m_tot))
L_indices = np.arange(len(L_tot))
m_tick_pos = np.arange(0, len(m_indices), 10)
L_tick_pos = np.arange(0, len(L_indices), 10)

# --- Step 2: file loader ---
def importfile_atmos_model(filename):
    """
    Load time, ocean, atmosphere series from text file.
    Assumes file has 3 columns (time, ocean, atmosphere) = (time, x1, x2) 
    Returns numpy arrays: time, ocean, atmosphere.
    """
    try:
        df = pd.read_csv(filename, delim_whitespace=True, header=0)
    except Exception as e:
        raise IOError(f"Could not read file {filename}: {e}")

    if df.shape[1] < 3:
        raise ValueError(f"File {filename} must have at least 3 columns, found {df.shape[1]}.")

    stop = 60000
    step = 1
    df = df.iloc[:stop:step, :]   # slicing with step

    time = df.iloc[:stop, 0].values
    ocean = df.iloc[:stop, 1].values
    atmosphere = df.iloc[:stop, 2].values

    # Validation: ensure arrays are not empty
    if len(time) == 0 or len(ocean) == 0 or len(atmosphere) == 0:
        raise ValueError(f"File {filename} produced empty arrays. Check file contents.")

    print("time, ocean, atmosphere shapes:", time.shape, ocean.shape, atmosphere.shape)
    return time, ocean, atmosphere


# --- Step 3: extract_te_variables (translation of MATLAB function) ---
def extract_te_variables(driver_series, target_series, m, L=1):
    """
    Inputs:
    %   driver_series - Time series of the driver variable (column vector)
    %   target_series - Time series of the target variable (column vector)
    %   m - Length of history (number of past time points to include)
    %   L - Time lag (number of time points to skip before starting AR model)
    %       Default: L = 1 (standard case, use immediately preceding values)
    %
    % Outputs:
    %   x - Present values of driver variable
    %   y - Present values of target variable
    %   z - Past values of target variable (m previous time points, starting L steps back)
    %
    % For Transfer Entropy computation using I(x,y|z) where:
    %   x = driver present
    %   y = target present  
    %   z = target past (history of length m, starting at lag L)
    %
    % Example: If L=3 and m=2, to predict time t, we use:
    %   - Present: driver(t) and target(t)
    %   - Past: target(t-4) and target(t-5) (skipping t-1, t-2, t-3)"""
    
    driver_series = np.asarray(driver_series).flatten()
    target_series = np.asarray(target_series).flatten()
    n = len(target_series)

    if m < 1 or m >= n:
        raise ValueError("m must be >= 1 and < length of time series")

    # Extract present values (from time point (m+L+1) to end)
    # This ensures we have m past values available starting L steps back
    x = driver_series[m+L:] # present driver values 
    y = target_series[m+L:] # present target values 

    # Extract past values of target variable with lag L
    # z will be a matrix where each row contains m past values
    # starting L time points before the present
    num_samples = n - m - L
    z = np.zeros((num_samples, m))

    for i in range(num_samples):
        # For sample i at time t=(m+L+i), collect m values starting at t-L-m
        # This means we skip L points (t-1, t-2, ..., t-L) and then take m points
        # start index i : corresponds to t-L-m in the original series 
        # end index i+m-1 : corresponds to t-L-1 in the original series
        z[i, :] = target_series[i:i+m]

    return x, y, z

# --- step 4: gmci has been imported from the file ---

# --- Step 5: main loop ---
def compute_te(filename):
    _, ocean, atmosphere = importfile_atmos_model(filename)
    TE_O_to_A = np.zeros((len(m_tot), len(L_tot)))
    TE_A_to_O = np.zeros((len(m_tot), len(L_tot)))

    total_iters = len(m_tot) * len(L_tot)

    with tqdm(total=total_iters, desc="Computing TE") as pbar:
        for i, m in enumerate(m_tot):
            for j, L in enumerate(L_tot):
                # O → A
                d, t = ocean, atmosphere
                x, y, z = extract_te_variables(d, t, m, L)
                x, y, z = x.reshape(1, -1), y.reshape(1, -1), z.T
                TE_O_to_A[i, j] = gccmi_ccc(x, y, z)

                # A → O
                d, t = atmosphere, ocean
                x, y, z = extract_te_variables(d, t, m, L)
                x, y, z = x.reshape(1, -1), y.reshape(1, -1), z.T
                TE_A_to_O[i, j] = gccmi_ccc(x, y, z)

                pbar.update(1)

    return TE_O_to_A, TE_A_to_O

# --- Step 6: plotting (example for one file) ---
#TE_O_to_A, TE_A_to_O = compute_te("../lin_oscillator/2D_system_data/2D_timeseries_bias1.txt")

"""
Computing for different mu values
"""
files = ["../lin_oscillator//data_lin_oscillator/data_mu0.txt", 
         "../lin_oscillator//data_lin_oscillator/data_mu0.001.txt",
         "../lin_oscillator//data_lin_oscillator/data_mu0.01.txt"]
mu = 0
for file in files: 
    TE_O_to_A, TE_A_to_O = compute_te(file)
    # Save as compressed NumPy archive (best for large arrays)
    np.savez(f"te_results_mu{mu}_notdownsampled.npz", 
            TE_O_to_A=TE_O_to_A, 
            TE_A_to_O=TE_A_to_O, 
            m_tot=m_tot, 
            L_tot=L_tot)
    mu = mu + 1
    