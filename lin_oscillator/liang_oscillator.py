# here goes the code to apply liang to this 

import sys
import numpy as np
import matplotlib

sys.path.insert(0, '/home/chiaraz/Liang_Index_climdyn')
from function_liang import compute_liang

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

    # empty arrays for time series from file 
    data_in_file = []

    with open(f"data_mu{mu}.txt", 'r') as file: 
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
    print(np.shape(data_in_file))

    print(data_in_file[0], ": first column?")

    results = compute_liang(data_in_file[0], data_in_file[1], 1, 100)
    print("T21,tau21,error_T21,error_tau21,R,error_R,error_T21_FI")
    print(results)
    print("tau_12: ", results[1], " +/- ", results[3])


def te_biv_compute_save(mu): 
    import os
    from jpype.types import JArray, JDouble

    # 0. Load/prepare the data:
    data = []

    with open(f"data_mu{mu}.txt", 'r') as file: 
        for index, line in enumerate(file):
            if index == 0:
                continue  # Skip header
            try:
                row = [float(val) for val in line.strip().split(',')]
                if len(row) < 3:
                    continue
                # Only take the second and third columns
                data.append([row[1], row[2]])
            except ValueError:
                continue

    # Convert to numpy array and transpose: shape (n_variables x time_steps)
    data = np.transpose(np.array(data))  # shape: 2 x N

    # 1. Construct the calculator:
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    calc = calcClass()

    # 2. Set properties (as needed)
    calc.setProperty("k", "2")
    calc.setProperty("AUTO_EMBED_RAGWITZ_NUM_NNS", "4")

    # Create results directory if needed
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 3. Open a CSV file to store results
    with open(os.path.join(results_dir, f"results_mu{mu}.csv"), mode="w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            "Source", "Destination", "TE_Kraskov", "Null Mean", "Null Std Dev", "P-value", "Surrogates"
        ])

        # 4. Loop through pairs of variables
        for s in range(2):
            for d in range(2):
                if s == d:
                    continue  # skip self-transfer

                source = JArray(JDouble, 1)(data[s, :].tolist())
                destination = JArray(JDouble, 1)(data[d, :].tolist())

                # 5. Initialize calculator and compute TE
                calc.initialise()
                calc.setObservations(source, destination)
                result = calc.computeAverageLocalOfObservations()

                # 6. Compute statistical significance
                measDist = calc.computeSignificance(100)

                print(
                    f"TE_Kraskov (col_{s} -> col_{d}) = {result:.4f} nats "
                    f"(null: {measDist.getMeanOfDistribution():.4f} ± {measDist.getStdOfDistribution():.4f}; "
                    f"p = {measDist.pValue:.5f})"
                )

                csvwriter.writerow([
                    s, d, result, measDist.getMeanOfDistribution(),
                    measDist.getStdOfDistribution(), measDist.pValue, 100
                ])


te_biv_compute_save(100)

#liang_compute_save(0)
#liang_compute_save(100)