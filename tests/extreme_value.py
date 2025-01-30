import numpy as np 
import matplotlib.pyplot as plt 

import sys 
sys.path.insert(0, '/home/chiaraz/thesis')

from functions_for_maooam import is_extreme_value

# Example usage:
data = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,20,400,59]
extreme_mask = is_extreme_value(data,0.99)
print(extreme_mask)  # True for the top 5% of values