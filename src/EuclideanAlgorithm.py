import numpy as matrix

row = 256
col = 256

s = [[0 for i in range(col)] for j in range(row)]

def mean(array):
    sum = 0
    meanmat = [[0 for i in range(col)] for j in range(row)]
    
    for i in range (0,256):
        for j in range (0,256):
            for matrix in array:
                meanmat[i][j] += matrix[i][j]
            meanmat[i][j] /= 256

    return meanmat

def diff(array):
    for i in range (row):
        for j in range (col):
            for matrix in array:
                matrix[i][j] -= mean(array)

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

def covarian(mtr):
    tmatrix = matrix.transpose(mtr)
    hasilMatrix = matrix.multiply(mtr,tmatrix)
    return hasilMatrix


mean = mean(s)
diff(s)
covar = covarian(s)