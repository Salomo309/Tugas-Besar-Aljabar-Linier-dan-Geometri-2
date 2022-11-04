import numpy as matrix

row = 256
col = 256

s = [[0 for i in range(col)] for j in range(row)]

def mean(array):
    sum = 0
    meanmat = [[0 for i in range(col)] for j in range(row)]
    
    for i in range (0,256):
        for j in range (0,256):
            for mat in array:
                meanmat[i][j] += mat[i][j]
            meanmat[i][j] /= 256

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
    for mat in array:
        matrix.concatenate(mat)

    pass

def covarian(mtr):
    tmatrix = matrix.transpose(mtr)
    hasilMatrix = matrix.multiply(mtr,tmatrix)
    return hasilMatrix


mean = mean(s)
diff(s)
covar = covarian(s)