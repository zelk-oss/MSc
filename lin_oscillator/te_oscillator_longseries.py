""" Batch transfer entropy analysis of linear oscillators system """

from jpype import *
import numpy as np
import os

# Path to JIDT jar
jarLocation = "/mnt/c/Users/zelco/Documents/JIDT/infodynamics.jar"

# Start JVM
if not isJVMStarted():
    startJVM(getDefaultJVMPath(), "-ea", f"-Djava.class.path={jarLocation}", convertStrings=True)

def compute_TE(ocean, atmosphere):
    ocean_java = JArray(JDouble, 1)(ocean.tolist())
    atmosphere_java = JArray(JDouble, 1)(atmosphere.tolist())
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    calc = calcClass()
    calc.setProperty("k", "10")

    # Ocean → Atmosphere
    calc.initialise()
    calc.setObservations(ocean_java, atmosphere_java)
    te_oa = calc.computeAverageLocalOfObservations()
    err_oa = calc.computeSignificance(0).getStdOfDistribution()

    # Atmosphere → Ocean
    calc.initialise()
    calc.setObservations(atmosphere_java, ocean_java)
    te_ao = calc.computeAverageLocalOfObservations()
    err_ao = calc.computeSignificance(0).getStdOfDistribution()

    return te_oa, err_oa, te_ao, err_ao

# Prepare output path
output_path = os.path.expanduser("~/results/te_results_mu0.txt")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Open output file
with open(output_path, "w") as out_file:
    out_file.write("Npoints TE_OA ±err TE_AO ±err\n")

    point_counts = np.logspace(3, 5, 12, dtype=int)
    for N in point_counts:
        filename = os.path.expanduser(f"~/data_thesis/lin_oscillator/data_mu0.1_n{N}.txt")
        if not os.path.isfile(filename):
            print(f"Warning: {filename} not found.")
            continue

        data = np.loadtxt(filename, skiprows=1)
        ocean = data[:, 1]
        atmosphere = data[:, 2]

        te_oa, err_oa, te_ao, err_ao = compute_TE(ocean, atmosphere)

        print(f"{N}: TE_OA = {te_oa:.4f} ± {err_oa:.4f} | TE_AO = {te_ao:.4f} ± {err_ao:.4f}")
        out_file.write(f"{N} {te_oa:.6f} {err_oa:.6f} {te_ao:.6f} {err_ao:.6f}\n")
