import numpy as np 
import sys 

v = np.zeros((36,3000))
print(np.shape(v))

for i in range (0,36):
    for j in range (0,3000): 
        v[i,j] = i+j


sys.path.insert(0, '/home/chiaraz/thesis')
from functions_for_maooam import introduce_lag_fourier

laggedv = introduce_lag_fourier(v, -300, True)

print(v[0])
print(v[19])
print(v[20])
print("ecco le nuove")
print(laggedv[0])
print(laggedv[19])
print(laggedv[20])