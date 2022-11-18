import numpy as np


def qr(A):
    ''' QR DECOMPOSITION USING SCHWARZ-RUTISHAUSER ALGORITHM
        SOURCE: https://towardsdatascience.com/can-qr-decomposition-be-actually-faster-schwarz-rutishauser-algorithm-a32c0cde8b9b
    '''
    # get shape
    _, n = np.shape(A)

    Q = A
    R = np.zeros((n, n))

    for k in range(n):
        for i in range(k):
            R[i, k] = np.transpose(Q[:, i]).dot(Q[:, k])
            Q[:, k] = Q[:, k] - (R[i, k] * Q[:, i])

        R[k, k] = np.linalg.norm(Q[:, k])
        Q[:, k] = Q[:, k] / R[k, k]

    return Q, R


def qr_iteration(C1, k):
    ''' SIMULTANEOUS QR ITERATION FOR EIGENVALUE R AND EIGENVECTOR Q (FROM REDUCED COVARIANCE MATRIX)
        k is the amount of eigenvector we want (the best eigenvalues)
        usually k = 0.1 M because 90% of the total variance is contained in the first 5% to 10% eigenvectors
        SOURCE: https://www.researchgate.net/publication/260880521
    '''
    n, m = C1.shape
    Q = np.random.rand(n, k)
    Q, _ = qr(Q)

    for i in range(1000):
        Z = C1 @ Q
        Q, R = qr(Z)

    return np.diag(R), Q  # dim of R: k x k, dim of Q: n x k


def eigen_vector(A, v):
    ''' RETURN THE EIGEN VECTOR OF COVARIANCE MATRIX
        if covariance matrix C = A.A^T has dimension of 2048 x 2048 and C1 = A^T.A has dimension of n x n
        then eig vectors e_i of C is given by A.v_i where v_i is eig vectors of C1
        SOURCE: https://www.researchgate.net/publication/260880521
    '''
    E = np.zeros((A.shape[0], v.shape[1]))
    for i in range(v.shape[1]):
        E[:, i] = (A.dot(v[:, i]))

    return E  # 2048 x d, where d is the desired amount of eigen vectors we want to find


def proj(E, A):
    '''
    PROJECTION OF DATA MATRIX A AND CALCULATION OF y_i VECTORS OF MATRIX Y =(y1,...,yM )
    OR EIGEN FACE
    '''
    return np.transpose(E).dot(A)  # d x m

# def qr(A):
#     # get shape
#     A = np.array(A, dtype=np.double)
#     _, n = np.shape(A)

#     Q = np.array(A, dtype=np.double)
#     R = np.zeros((n, n), dtype=np.double)

#     for k in range(n):
#         for i in range(k):
#             R[i, k] = np.transpose(Q[:, i]).dot(Q[:, k])
#             Q[:, k] = Q[:, k] - R[i, k] * Q[:, i]

#         R[k, k] = np.linalg.norm(Q[:, k])
#         Q[:, k] = Q[:, k] / R[k, k]

#     return -Q, -R


# def qr_iteration(A):

#     # Algorithm to find eigenValues and eigenVector matrix using simultaneous power iteration.

#     n, m = A.shape
#     Q = np.random.rand(n, m)  # Make a random n x k matrix
#     Q, _ = np.linalg.qr(Q)  # Use np.linalg.qr decomposition to Q

#     for i in range(1000):
#         Z = A.dot(Q)
#         Q, R = np.linalg.qr(Z)
#     # Do the same thing over and over until it converges
#     return np.diag(R), Q


# def eigen_value(A):
#     A_k = np.array(A, dtype=np.double)
#     # A_k = hessenberg(A_k, calc_q=False)
#     for i in range(1000):
#         Q, R = qr(A_k)
#         A_k = np.dot(R, Q)
#     return np.flip(np.sort(np.diag(A_k)))


# def eigen_vector(A, eig, cov):

#     n = cov.shape[0]
#     I = np.eye(n, dtype=np.double)
#     eig_vec = []
#     eig_v = []
#     for i in range(len(eig) // 10):
#         if (abs(eig[i]) < 0.0000001):
#             continue
#         copy = np.array(cov, dtype=np.double)
#         tes = np.subtract(np.multiply(I, eig[i]), copy)
#         # aug = np.concatenate((copy, b), axis=1)
#         # aug = Matrix(aug)
#         # print(aug)
#         print(null_space(tes))
#         v_i = np.transpose(null_space(tes))
#         print("null")
#         if (len(v_i) == 0):
#             v_i = [0 for j in range(n)]
#         else:
#             v_i = v_i[0]
#             # eig_v.append(v_i)
#         u_i = np.matmul(A, v_i)
#         # print(u_i)
#         eig_vec.append(u_i)
#     # print(eig_v)
#     return np.transpose(np.array(eig_vec, dtype=np.double))


# def eigen_face(eig_vec, sub):
#     # print(v.shape)
#     # print(sub_i.shape)
#     return np.matmul(np.transpose(eig_vec), sub)

# Tempat pembuangan kode

# def eigen_value_with_shift(A):
#     A_k = np.array(A, dtype=np.float32)
#     n = len(A_k)
#     I = np.eye(n, dtype=np.float32)
#     # shifting algorithm
#     A_k = hessenberg(A_k, calc_q=False)
#     s_k = A_k[n-1, n-1]
#     shift = s_k * I
#     for i in range(10000):
#         Q, R = qr(np.subtract(A_k, shift))
#         A_k = np.add(np.dot(R, Q), shift)
#     return np.sort(np.diag(A_k))

# def forward_sub_zeros(L):
#     # number of solutions
#     n = L.shape[0]
#     # initialize solution vector
#     y = np.zeros(n, dtype=np.double)

#     b = 0
#     for k in range(n):
#         for i in range(k):
#             b -= L[k, i] * y[i]
#     return y


# def backward_sub_zeros(U):
#     # number of solutions
#     n = U.shape[0]
#     # initialize solution vector
#     x = np.zeros(n, dtype=np.double)

#     for k in range(n):
#         x[k] = U[k, k]
#         for i in range(k):
#             x[k] -= U[k, i]

#     return x


# def sum_for_crout(L, U, k, i, j):
#     sum = 0

#     for m in range(0, k):
#         sum += L[i, m] * U[m, j]
#     return sum


# def crout(A):
#     n = A.shape[0]
#     # initialize L and U
#     L = np.zeros((n, n), dtype=np.double)
#     U = np.zeros((n, n), dtype=np.double)

#     for k in range(n):  # k = 1

#         L[k, k] = 1
#         for i in range(k, n):  # range(1, 3)
#             # U[1, 1] = A[2, 1] -
#             # U[1, 2]
#             U[k, i] = A[k, i] - sum_for_crout(L, U, k, k, i)
#         for j in range(k + 1, n):  # range(2, 3)
#             # L[2, 1]
#             L[j, k] = (A[j, k] - sum_for_crout(L, U, k, j, k)) / U[k, k]
#             # L[j, k] /= U[k, k]

#     return L, U


# def getLeadColIdx(A, r):
#     _, nc = A.shape
#     for i in range(nc):
#         if abs(A[r, i]) > 0:
#             return i

#     return 999999999999


# def pivot(A, startRow):
#     nr, _ = A.shape
#     leadr, leadc = startRow, getLeadColIdx(A, startRow)
#     for r in range(startRow + 1, nr):
#         lead = getLeadColIdx(A, r)
#         if (lead < leadc):
#             leadr, leadc = r, lead
#     return leadr, leadc


# def RREF(A):
    # nr, nc = A.shape
    # pr, pc = pivot(A, 0)
    # for r in range(nr - 1):

    #     # if pivot is not element of vector b
    #     if (pc < nc - 1):
    #         # if pivot's row is not where it suppossed to be
    #         if (pr != r):
    #             A[[pr, r]] = A[[r, pr]]  # swap it
    #         # if pivot element is not 1
    #         if A[r, pc] != 1:
    #             # convert it to 1
    #             mul = 1 / A[r, pc]
    #             A[r, :] *= mul

    #         # make zeros for all rows beside pr in the same column
    #         for x in range(r, nr):
    #             # print(x)
    #             # print(A)
    #             if (r != x):
    #                 if (A[x, pc] != 0):
    #                     mul = A[x, pc] / A[r, pc]
    #                     A[x, :] -= A[r, :] * mul
    #                     # print(A)

    #     elif pc == nc - 1:
    #         break
    #     pr, pc = pivot(A, r + 1)

    # # last iter
    # if (pc < nc - 1 and A[nr - 1, pc] != 1):
    #     if A[nr - 1, pc] != 1:
    #         # convert it to 1
    #         mul = 1 / A[nr - 1, pc]
    #         for c in range(nc):
    #             A[nr - 1, c] *= mul

    #     # for x in range(nr - 1):
    #     #     # print(x)
    #     #     # print(A)
    #     #     if (A[x, pc] != 0):
    #     #         mul = A[x, pc] / A[nr-1, pc]
    #     #         for c in range(nc):
    #     #             A[x, c] -= A[nr-1, c] * mul

    # return A
