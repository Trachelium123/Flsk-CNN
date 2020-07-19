import sys


#
#
# def func(a, b):
#     return (a + b)
#
#
# def getarr(arr):
#     print(arr)


# if __name__ == '__main__':
#     # print(float(sys.argv))
# #
#     print(func(a[0], a[1]))
import numpy as np
arrs = np.random.random((100, 24))
with open('text.txt', 'a')as f:
    for i in range(arrs.shape[0]):
        for j in range(arrs.shape[1]):
            f.write(str(arrs[i][j]))
            f.write(',')
        f.write('\n')
