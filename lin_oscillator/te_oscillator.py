""" transfer entropy analysis of linear oscillators system 
"""

from jpype import *
import numpy as np
import sys

# Use the correct path under WSL to access the JIDT jar on Windows
jarLocation = "/mnt/c/Users/zelco/Documents/JIDT/infodynamics.jar"

# Start the JVM if not already started
if not isJVMStarted():
    startJVM(getDefaultJVMPath(), "-ea", f"-Djava.class.path={jarLocation}", convertStrings=True)

def compute_TE_from_file(input_file_path):
    # Read the data (skip header)
    data = np.loadtxt(input_file_path, skiprows=1)
    
    # Extract only the ocean and atmosphere columns
    ocean = data[:, 1]
    atmosphere = data[:, 2]

    # Convert to Java arrays
    ocean_java = JArray(JDouble, 1)(ocean.tolist())
    atmosphere_java = JArray(JDouble, 1)(atmosphere.tolist())

    # Create TE calculator
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    calc = calcClass()

    # Optional: change nearest-neighbor parameter
    calc.setProperty("k", "2")
    
    # -- Ocean → Atmosphere
    calc.initialise()
    calc.setObservations(ocean_java, atmosphere_java)
    te_oa = calc.computeAverageLocalOfObservations()
    dist_oa = calc.computeSignificance(100)
    err_oa = dist_oa.getStdOfDistribution()

    print(f"TE (Ocean → Atmosphere) [{input_file_path}]: %.4f ± %.4f nats" % (te_oa, err_oa))

    # -- Atmosphere → Ocean
    calc.initialise()
    calc.setObservations(atmosphere_java, ocean_java)
    te_ao = calc.computeAverageLocalOfObservations()
    dist_ao = calc.computeSignificance(100)
    err_ao = dist_ao.getStdOfDistribution()

    print(f"TE (Atmosphere → Ocean) [{input_file_path}]: %.4f ± %.4f nats" % (te_ao, err_ao))

# Example usage
compute_TE_from_file("data_mu0.01_8000.txt")
