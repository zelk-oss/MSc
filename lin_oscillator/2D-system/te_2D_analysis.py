"""
Analyze saved time series using Transfer Entropy.
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

# === Transfer Entropy analysis (JIDT) ===
jarLocation = "/mnt/c/Users/zelco/Documents/JIDT/infodynamics.jar"
if not isJVMStarted():
    startJVM(getDefaultJVMPath(), "-ea", f"-Djava.class.path={jarLocation}", convertStrings=True)

ocean_java = JArray(JDouble, 1)(X1.tolist())
atmo_java = JArray(JDouble, 1)(X2.tolist())
calcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
calc = calcClass()
# CHANGE EMBEDDING DIMENSION HERE
k = 20
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

print("\n=== Transfer Entropy (TE) ===\n")
print(f"TE (X1 → X2): {te_oa:.6g} nats, Dist mean: {dist_oa.getMeanOfDistribution():.6g}, "
      f"Dist std: {dist_oa.getStdOfDistribution():.6g}, p-value: {dist_oa.pValue:.6g}, "
      f"N surrogates: {Nsurrogates}")

print(f"TE (X2 → X1): {te_ao:.6g} nats, Dist mean: {dist_ao.getMeanOfDistribution():.6g}, "
      f"Dist std: {dist_ao.getStdOfDistribution():.6g}, p-value: {dist_ao.pValue:.6g}, "
      f"N surrogates: {Nsurrogates}")

# Save to custom directory
output_dir = os.path.expanduser("~/thesis/lin_oscillator/2D_system_data")
os.makedirs(output_dir, exist_ok=True)
filename = f"te_2D_embedding{k}_bias1.txt"
with open(os.path.join(output_dir, filename), 'w') as file: 
      file.write("=== Transfer Entropy (TE) ===")
      file.write(f"TE (X1 → X2): {te_oa:.6g} nats, Dist mean: {dist_oa.getMeanOfDistribution():.6g}, \n"
            f"Dist std: {dist_oa.getStdOfDistribution():.6g}, p-value: {dist_oa.pValue:.6g}, \n"
            f"N surrogates: {Nsurrogates} \n")

      file.write(f"TE (X2 → X1): {te_ao:.6g} nats, Dist mean: {dist_ao.getMeanOfDistribution():.6g}, \n"
            f"Dist std: {dist_ao.getStdOfDistribution():.6g}, p-value: {dist_ao.pValue:.6g}, \n"
            f"N surrogates: {Nsurrogates} \n")
print("saved file")