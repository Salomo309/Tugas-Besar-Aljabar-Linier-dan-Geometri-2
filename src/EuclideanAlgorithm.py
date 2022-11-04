import numpy as matrix

row = 5
col = 5

s = [[0 for i in range(col)] for j in range(row)]


def mean(array):
    sum = 0
    meanmat = [[0 for i in range(col)] for j in range(row)]

    for i in range(row):
        for j in range(col):
            for mat in array:
                meanmat[i][j] += mat[i][j]
            meanmat[i][j] = int(meanmat[i][j] / len(array))

    return meanmat


def diff(array):
    for mat in array:
        mat -= mean(array)

    return array

# def transpose(matrix):
#     for i in range (row):
#         for j in range (col):
#             matrix[i][j] = matrix[j][i]

# def multiplyMatrix(matrix1, matrix2):
#     hmatrix = [[0 for i in range(col)] for j in range(row)]

#     for i in range (row):
#         for j in range (col):
#             hmatrix[i][j] += matrix1[i][j] * matrix2[j][i]


def concat(array):
    mat = array[0]

    for i in range(1, len(array)):
        mat = matrix.concatenate((mat, array[i]), axis=1)
    return mat


def covarian(mtr):
    tmatrix = matrix.transpose(mtr)
    hasilMatrix = matrix.matmul(mtr, tmatrix)
    return hasilMatrix


# mean = mean(s)
# diff(s)
# concat(s)
# covar = covarian(s)
