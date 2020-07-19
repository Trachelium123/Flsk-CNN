import numpy as np


# 归一化
def maxminnorm(array):
    max = array.max()
    min = array.min()
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    norm = np.empty((data_rows, data_cols))
    for i in range(data_rows):
        for j in range(data_cols):
          norm[i, j] = (array[i, j] - min) / (max- min)
    return norm