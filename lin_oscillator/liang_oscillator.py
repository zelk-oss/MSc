# here goes the code to apply liang to this 

import sys
import numpy as np
import matplotlib

sys.path.insert(0, '/home/chiaraz/Liang_Index_climdyn')
from function_liang_nvar import compute_liang_nvar

from jpype import *
import numpy
import sys
import csv  # For writing results to a CSV file

# Our python data file readers are a bit of a hack, python users will do better on this:
sys.path.append("/mnt/c/Users/zelco/Documents/JIDT/demos/python")
import readFloatsFile

if not isJVMStarted():
    # Add JIDT jar library to the path
    jarLocation = "/mnt/c/Users/zelco/Documents/JIDT/infodynamics.jar"
    # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, convertStrings=True)

# ######################################
# functions for liang and TE computation
# ###################################### 

def liang_compute_save(mu):
    """ extract data from timeseries file. apply liang to all combinations of the two variables 
    print the results to terminal """

    # empty arrays for time series from file 
    data_in_file = []

    with open(f"/home/chiaraz/data_thesis/lin_oscillator/data_mu{mu}.txt", 'r') as file: 
        for index, line in enumerate(file):
            if index == 0: 
                continue  
            # handling potential invalid data 
            try:
                # turns lines in the file in tuples of 2 elements
                row = [float(value) for value in line.split()]
                if len(row) < 2: 
                    print("Line has insufficient data: {line.strip()}")
                    continue 
                # here we can do stuff with the data 
                data_in_file.append([row[1], row[2]])

            except ValueError:
                print(f"Could not convert line to floats: {line}")
    
    data_in_file = np.transpose(data_in_file)
    #print("shape of data in file: ", np.shape(data_in_file))
    X1 = data_in_file[0] # ocean 
    X2 = data_in_file[1] # atmosphere 
    print(np.shape(X1), np.shape(X2))

    # === Liang index analysis ===
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
    
    print(f"Results for mu={mu}")
    print("=== Liang Index ===")
    print("T matrix:\n", T)
    print("Significance (T):\n", sig_T)
    print("tau matrix:\n", tau)
    print("Significance tau:\n", sig_tau)
    print("R matrix:\n", R)
    print("Significance R:\n", sig_R)




def te_biv_compute_save(mu, k): 
    import os
    from jpype.types import JArray, JDouble

    # 0. Load/prepare the data:
    data = []

    with open(f"/home/chiaraz/data_thesis/lin_oscillator/data_mu{mu}.txt", 'r') as file: 
        for index, line in enumerate(file):
            if index == 0: 
                continue  
            # handling potential invalid data 
            try:
                # turns lines in the file in tuples of 2 elements
                row = [float(value) for value in line.split()]
                if len(row) < 2: 
                    print("Line has insufficient data: {line.strip()}")
                    continue 
                # here we can do stuff with the data 
                data.append([row[1], row[2]])

            except ValueError:
                print(f"Could not convert line to floats: {line}")

    # Convert to numpy array and transpose: shape (n_variables x time_steps)
    data = np.transpose(np.array(data))  # shape: 2 x N
    X1 = data[0] # ocean 
    X2 = data[1] # atmosphere 

    # === Transfer Entropy analysis (JIDT) ===
    jarLocation = "/mnt/c/Users/zelco/Documents/JIDT/infodynamics.jar"
    if not isJVMStarted():
        startJVM(getDefaultJVMPath(), "-ea", f"-Djava.class.path={jarLocation}", convertStrings=True)

    ocean_java = JArray(JDouble, 1)(X1.tolist())
    atmo_java = JArray(JDouble, 1)(X2.tolist())
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    calc = calcClass()
    # CHANGE EMBEDDING DIMENSION HERE
    calc.setProperty("k", f"{k}")
    Nsurrogates = 200

    print("starting computation...")
    # X1 → X2
    calc.initialise()
    calc.setObservations(ocean_java, atmo_java)
    te_oa = calc.computeAverageLocalOfObservations()
    dist_oa = calc.computeSignificance(Nsurrogates)

    # X2 → X1
    calc.initialise()
    calc.setObservations(atmo_java, ocean_java)
    te_ao = calc.computeAverageLocalOfObservations()
    dist_ao = calc.computeSignificance(Nsurrogates)

    print(f"Results for mu = {mu} ; k = {k}")
    print("=== Transfer Entropy (TE) ===")
    print(f"TE (X1 → X2): {te_oa:.6g} nats, Dist mean: {dist_oa.getMeanOfDistribution():.6g}, "
        f"Dist std: {dist_oa.getStdOfDistribution():.6g}, p-value: {dist_oa.pValue:.6g}, "
        f"N surrogates: {Nsurrogates}")

    print(f"TE (X2 → X1): {te_ao:.6g} nats, Dist mean: {dist_ao.getMeanOfDistribution():.6g}, "
        f"Dist std: {dist_ao.getStdOfDistribution():.6g}, p-value: {dist_ao.pValue:.6g}, "
        f"N surrogates: {Nsurrogates}", "\n")


te_biv_compute_save(1, 4)
te_biv_compute_save(1, 8)
te_biv_compute_save(10, 4) 
te_biv_compute_save(10, 8)

#liang_compute_save()
