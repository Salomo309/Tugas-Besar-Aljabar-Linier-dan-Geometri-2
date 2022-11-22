import numpy as np
import math


def norm(a):
    sum = 0
    b = np.transpose(a)
    for i in range(a.shape[0]):
        sum += b[i] * b[i]
    return math.sqrt(sum)


def mean_mat(arr):
    ''' MEAN OF MATRIX arr (2048 x M) where M is the total amount of pictures in dataset '''
    n = arr.shape[1]
    sum_mat = np.array(arr[:, 0])
    print(sum_mat.shape)
    for i in range(1, n):
        sum_mat += arr[:, i]
    print(sum_mat.shape)
    return sum_mat / n  # 2048 x 1


def sub_mat(arr, mean):
    ''' SUBSTITUTE MEAN (2048 x 1) ON EACH FLATTENED VECTORS IN ARR (2048 x M) 
        Result is mean face
    '''
    print(arr.shape)
    n = arr.shape[1]

    print(arr.shape)
    for i in range(n):
        arr[:, i] = np.subtract(arr[:, i], mean)

    return arr  # 2048 x m


def covariant(A):
    ''' COVARIANT OF MATRIX A (2048 X M) '''
    return np.matmul(np.transpose(A), A)  # m x m


def normalize(a):
    '''
        RETURN NORMALIZED OF VECTOR a
    '''
    a = np.divide(a, norm(a))
    return a


def euc_distance(a, b):
    '''
        RETURN EUCLEDIAN DISTANCE OF TWO VECTOR
    '''
    return norm(normalize(a) - normalize(b))
