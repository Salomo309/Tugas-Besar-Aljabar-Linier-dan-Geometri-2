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

# import numpy as matrix

# row = 5
# col = 5

# s = [[0 for i in range(col)] for j in range(row)]


# def mean(array):
#     sum = 0
#     meanmat = [[0 for i in range(col)] for j in range(row)]

#     for i in range(row):
#         for j in range(col):
#             for mat in array:
#                 meanmat[i][j] += mat[i][j]
#             meanmat[i][j] = int(meanmat[i][j] / len(array))

#     return meanmat


# def diff(array):
#     for mat in array:
#         mat -= mean(array)

#     return array

# # def transpose(matrix):
# #     for i in range (row):
# #         for j in range (col):
# #             matrix[i][j] = matrix[j][i]

# # def multiplyMatrix(matrix1, matrix2):
# #     hmatrix = [[0 for i in range(col)] for j in range(row)]

# #     for i in range (row):
# #         for j in range (col):
# #             hmatrix[i][j] += matrix1[i][j] * matrix2[j][i]


# def concat(array):
#     mat = array[0]

#     for i in range(1, len(array)):
#         mat = matrix.concatenate((mat, array[i]), axis=1)
#     return mat


# def covarian(mtr):
#     tmatrix = matrix.transpose(mtr)
#     hasilMatrix = matrix.matmul(mtr, tmatrix)
#     return hasilMatrix


# mean = mean(s)
# diff(s)
# concat(s)
# covar = covarian(s)
