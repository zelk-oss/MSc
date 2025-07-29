# we compute the active iformation storage as a function of the emedding history length "k"
# reference to https://github.com/jlizier/jidt/blob/master/demos/octave/SchreiberTransferEntropyExamples/README-SchreiberTeDemos.pdf


# compute_ais.py

from jpype import *
import numpy as np
import sys
import os

# Append your custom Python reader for JIDT
sys.path.append("/mnt/c/Users/zelco/Documents/JIDT/demos/python")
import readFloatsFile

def compute_AIS_over_range(file_path, max_k=15, out_file="ais_results.txt"):
    if not isJVMStarted():
        jarLocation = "/mnt/c/Users/zelco/Documents/JIDT/infodynamics.jar"
        startJVM(getDefaultJVMPath(), "-ea", f"-Djava.class.path={jarLocation}", convertStrings=True)

    # Load and convert data
    dataRaw = readFloatsFile.readFloatsFile(file_path)
    data = np.array(dataRaw)
    
    with open(out_file, "w") as f_out:
        f_out.write("k_history\tAIS_col1\tAIS_col2\n")
        for k_history in range(1, max_k + 1):
            ais_values = []
            for col in [1, 2]:  # column 2 and column 3
                variable = JArray(JDouble, 1)(data[:, col].tolist())

                # AIS Calculator
                calcClass = JPackage("infodynamics.measures.continuous.kraskov").ActiveInfoStorageCalculatorKraskov
                calc = calcClass()
                calc.setProperty("k_HISTORY", str(k_history))
                calc.setProperty("k", "10")
                calc.initialise()
                calc.setObservations(variable)

                result = calc.computeAverageLocalOfObservations()
                ais_values.append(result)
            
            f_out.write(f"{k_history}\t{ais_values[0]:.6f}\t{ais_values[1]:.6f}\n")
            print(f"k={k_history} -> AIS_col1={ais_values[0]:.4f}, AIS_col2={ais_values[1]:.4f}")

if __name__ == "__main__":
    file_path = "/home/chiaraz/thesis/lin_oscillator/2D_system_data/2D_timeseries.txt"
    compute_AIS_over_range(file_path)

