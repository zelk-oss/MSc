from jpype import *
import numpy
import sys
# Our python data file readers are a bit of a hack, python users will do better on this:
sys.path.append("/mnt/c/Users/zelco/Documents/JIDT/demos/python")
import readFloatsFile

if not isJVMStarted():
    # Add JIDT jar library to the path
    jarLocation = "/mnt/c/Users/zelco/Documents/JIDT/infodynamics.jar"
    # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, convertStrings=True)

import csv
import numpy as np

with open("Normalized_Climate_anomalies.csv", 'r') as file:
    reader = csv.reader(file)
    header = next(reader)   # ['EPAC','CPAC','NAT','WPAC']
    data = []
    for row in reader:
        try:
            values = [float(x) for x in row]
            data.append(values)
        except ValueError:
            print(f"Skipping bad line: {row}")

data = np.array(data)   # shape (N, 4)

# 1. Construct the calculator:
calcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
calc = calcClass()
# 2. Set any properties to non-default values:
# No properties were set to non-default values

# Compute for all pairs:
for s in range(4):
    for d in range(4):
        # For each source-dest pair:
        if (s == d):
            continue
        source = JArray(JDouble, 1)(data[:, s].tolist())
        destination = JArray(JDouble, 1)(data[:, d].tolist())

        # 3. Initialise the calculator for (re-)use:
        calc.initialise()
        # 4. Supply the sample data:
        calc.setObservations(source, destination)
        # 5. Compute the estimate:
        result = calc.computeAverageLocalOfObservations()
        # 6. Compute the (statistical significance via) null distribution empirically (e.g. with 100 permutations):
        measDist = calc.computeSignificance(1000)

        print("TE_Kraskov (KSG)(col_%d -> col_%d) = %.4f nats (null: %.4f +/- %.4f std dev.; p(surrogate > measured)=%.5f from %d surrogates)" %\
            (s, d, result, measDist.getMeanOfDistribution(), measDist.getStdOfDistribution(), measDist.pValue, 100))
