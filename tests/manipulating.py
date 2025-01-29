import numpy as np 

array = np.random.randint(0, 100, size=(4, 4, 4))
print(array.shape)
print("original array", array)

sampled = array[::2, 0,0]
print("sampled array = ", sampled)

reshaped = sampled.reshape(sampled.shape[0], -1)
print("reshaped = ", reshaped)
print(reshaped.shape)

print( np.random.randint(0,10, size=(10,1)))