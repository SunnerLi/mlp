import numpy as np

a = np.asarray(
    [[1, 2, 3]]
)
b = np.asarray(
    [[1], [1], [1]]
)
print(np.matmul(np.linalg.inv(a), np.matmul(a, b)))