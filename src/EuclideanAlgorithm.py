import numpy as matrix

row = 6
col = 6

s = [[0 for i in range(col)] for j in range(row)]

def mean(mtr):
    sum = 0

    for i in range (row):
        for j in range (col):
            sum += mtr[i][j]

    return sum/col

def diff(mtr):
    for i in range (row):
        for j in range (col):
            mtr[i][j] -= mean(mtr)

# def transpose(matrix):
#     for i in range (row):
#         for j in range (col):
#             matrix[i][j] = matrix[j][i]

# def multiplyMatrix(matrix1, matrix2):
#     hmatrix = [[0 for i in range(col)] for j in range(row)]

#     for i in range (row):
#         for j in range (col):
#             hmatrix[i][j] += matrix1[i][j] * matrix2[j][i]

def covarian(mtr):
    tmatrix = matrix.transpose(matrix)
    hasilMatrix = matrix.multiply(matrix,tmatrix)
    return hasilMatrix


mean = mean(s)
diff(s)
covar = covarian(s)