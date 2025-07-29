# computing range of TE values chaning nearest neighs from 1 to 15 with fixed value k_HIST = 3
# which is the optimal value computed with ais_k.py 

# compute_te_alg1_alg2.py

from jpype import *
import numpy as np
import sys
import os

# Path to the JIDT library and data
jar_path = "/mnt/c/Users/zelco/Documents/JIDT/infodynamics.jar"
data_file = "/home/chiaraz/thesis/lin_oscillator/2D_system_data/2D_timeseries.txt"

# Append your JIDT python readers
sys.path.append("/mnt/c/Users/zelco/Documents/JIDT/demos/python")
import readFloatsFile

def compute_te_alg(alg_num, out_filename):
    if not isJVMStarted():
        startJVM(getDefaultJVMPath(), "-ea", f"-Djava.class.path={jar_path}", convertStrings=True)

    data_raw = readFloatsFile.readFloatsFile(data_file)
    data = np.array(data_raw)
    
    col1 = JArray(JDouble, 1)(data[:, 1].tolist())  # second column (index 1)
    col2 = JArray(JDouble, 1)(data[:, 2].tolist())  # third column (index 2)

    calcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov

    with open(out_filename, "w") as f_out:
        f_out.write("k\tTE_1to2\tstd_1to2\tpval_1to2\tTE_2to1\tstd_2to1\tpval_2to1\n")

        for k_val in range(1, 16):  # k from 1 to 15
            def compute_te(source, dest):
                calc = calcClass()
                calc.setProperty("k_HISTORY", "3")
                calc.setProperty("l_HISTORY", "3")
                calc.setProperty("k", str(k_val))
                calc.setProperty("ALG_NUM", str(alg_num))
                calc.setProperty("AUTO_EMBED_RAGWITZ_NUM_NNS", "4")
                calc.initialise()
                calc.setObservations(source, dest)
                te = calc.computeAverageLocalOfObservations()
                dist = calc.computeSignificance(100)
                return te, dist.getStdOfDistribution(), dist.pValue

            te_1to2, std_1to2, pval_1to2 = compute_te(col1, col2)
            te_2to1, std_2to1, pval_2to1 = compute_te(col2, col1)

            print(f"ALG {alg_num} | k={k_val} | TE 1→2: {te_1to2:.4f} ± {std_1to2:.4f}, TE 2→1: {te_2to1:.4f} ± {std_2to1:.4f}")
            f_out.write(f"{k_val}\t{te_1to2:.6f}\t{std_1to2:.6f}\t{pval_1to2:.6f}\t{te_2to1:.6f}\t{std_2to1:.6f}\t{pval_2to1:.6f}\n")

if __name__ == "__main__":
    #compute_te_alg(1, "te_results_alg1.txt")
    compute_te_alg(2, "te_results_alg2.txt")
