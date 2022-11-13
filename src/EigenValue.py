import numpy as np
from scipy.linalg import null_space


def qr(A):
    # get shape
    A = np.array(A, dtype=np.double)
    _, n = np.shape(A)

    Q = np.array(A, dtype=np.double)
    R = np.zeros((n, n), dtype=np.double)

    for k in range(n):
        for i in range(k):
            R[i, k] = np.transpose(Q[:, i]).dot(Q[:, k])
            Q[:, k] = Q[:, k] - R[i, k] * Q[:, i]

        R[k, k] = np.linalg.norm(Q[:, k])
        Q[:, k] = Q[:, k] / R[k, k]

    return -Q, -R


def eigen_value(A):
    A_k = np.array(A, dtype=np.double)
    # A_k = hessenberg(A_k, calc_q=False)
    for i in range(10000):
        Q, R = qr(A_k)
        A_k = np.dot(R, Q)
    return np.flip(np.sort(np.diag(A_k)))


def eigen_vector(A, eig, cov):

    n = cov.shape[0]
    I = np.eye(n, dtype=np.double)
    eig_vec = []
    for i in range(len(eig) // 2):
        if (abs(eig[i]) < 0.0000001):
            continue
        copy = np.array(cov, dtype=np.double)
        tes = np.subtract(copy, np.multiply(I, eig[i]))
        # aug = np.concatenate((copy, b), axis=1)
        # aug = Matrix(aug)
        # print(aug)
        v_i = np.transpose(null_space(tes))
        if (len(v_i) == 0):
            v_i = [0 for j in range(n)]
        else:
            v_i = v_i[0]
        u_i = np.matmul(v_i, A)
        # print(u_i)
        eig_vec.append(u_i)

    return np.array(eig_vec, dtype=np.double)


def eigen_face(v, sub_i):
    # print(v.shape)
    # print(sub_i.shape)
    ef_i = np.matmul(v, sub_i)
    return ef_i

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
