import numpy as np


def mean_mat(arr):
    ''' MEAN OF MATRIX arr (2048 x M) where M is the total amount of pictures in dataset '''
    n = arr.shape[1]
    sum_mat = np.array(arr[:, 0])
    print(sum_mat.shape)
    for i in range(1, n):
        sum_mat += arr[:, i]

    return sum_mat / n  # 2048 x 1


def sub_mat(arr, mean):
    ''' SUBSTITUTE MEAN (2048 x 1) ON EACH FLATTENED VECTORS IN ARR (2048 x M) '''
    print(arr.shape)
    n = arr.shape[1]

    print(arr.shape)
    for i in range(n):
        arr[:, i] = np.subtract(arr[:, i], mean)

    return arr  # 2048 x m


def covariant(A):
    ''' COVARIANT OF MATRIX A (2048 X M) '''
    return np.matmul(np.transpose(A), A)  # m x m


def normalize(E):
    '''
        RETURN NORMALIZED OF VECTOR E
    '''
    E = np.divide(E, np.linalg.norm(E))

    return E  # 2048 x d


def euc_distance(a, b):
    '''
        RETURN EUCLEDIAN DISTANCE OF TWO VECTOR
    '''
    return np.linalg.norm(normalize(a) - normalize(b))
