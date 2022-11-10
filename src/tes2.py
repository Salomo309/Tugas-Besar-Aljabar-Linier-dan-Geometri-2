import numpy as np

A = np.array([[1, 1, 2], [2, 3, 1]])
B = A[:, 0]
np.transpose(B)
print(B)
