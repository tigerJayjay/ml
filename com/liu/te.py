import numpy as np
a = np.array(((1,2),(3,4)))
print(a.shape[1])
print(tuple(range(a.shape[1])))
print([tuple(range(a.shape[1]))])
b = [1,2,3]
print(b[-1])