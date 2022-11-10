import numpy as np
import EuclideanAlgorithm as eucl

# TODO: Givens Rotation

# TODO: Hessenberg Matrix Form


def qr(A):
    # get shape
    r, c = A.shape

    Q = np.array(A, dtype=np.double)
    R = np.zeros([r, c], dtype=np.double)

    for k in range(c):
        for i in range(k):
            R[i, k] = np.transpose(Q[:, i]).dot(Q[:, k])
            Q[:, k] = Q[:, k] - R[i, k] * Q[:, i]
        R[k, k] = np.linalg.norm(Q[:, k])
        Q[:, k] = Q[:, k] / R[k, k]

    return -Q, -R


# A = np.array([[1, 1, 2], [2, 3, 1], [2, 3, 1]])
# Q1, R1 = np.linalg.qr(A)
# Q2, R2 = schwarz_rutishauser(A)
# print(Q1)
# print(Q2)

# RECOMMENDED
def eigen_value_with_shift(A):
    A_k = np.array(A, dtype=np.double)
    n = len(A_k)
    I = np.eye(n)

    # shifting algorithm
    s_k = A_k[n-1][n-1]
    shift = s_k * I

    for i in range(100):
        Q, R = qr(np.subtract(A_k, shift))
        A_k = np.add(np.dot(R, Q), shift)
    return np.sort(np.diag(A_k))


def eigen_value(A):
    A_k = np.array(A, dtype=np.double)
    n = len(A_k)
    I = np.eye(n)

    for i in range(100):
        Q, R = qr(A_k)
        A_k = np.dot(R, Q)
        I = np.dot(I, Q)
    return np.sort(np.diag(A_k))


def forward_sub_zeros(L):
    # number of solutions
    n = L.shape[0]
    # initialize solution vector
    y = np.zeros(n, dtype=np.double)

    b = 0
    for k in range(n):
        for i in range(k):
            b -= L[k, i] * y[i]
    return y


def backward_sub_zeros(U):
    # number of solutions
    n = U.shape[0]
    # initialize solution vector
    x = np.zeros(n, dtype=np.double)

    for k in range(n):
        x[k] = U[k, k]
        for i in range(k):
            x[k] -= U[k, i]

    return x


def sum_for_crout(L, U, k, i, j):
    sum = 0

    for m in range(0, k):
        sum += L[i, m] * U[m, j]
    return sum


def crout(A):
    n = A.shape[0]
    # initialize L and U
    L = np.zeros((n, n), dtype=np.double)
    U = np.zeros((n, n), dtype=np.double)

    for k in range(n):  # k = 1

        L[k, k] = 1
        for i in range(k, n):  # range(1, 3)
            # U[1, 1] = A[2, 1] -
            # U[1, 2]
            U[k, i] = A[k, i] - sum_for_crout(L, U, k, k, i)
        for j in range(k + 1, n):  # range(2, 3)
            # L[2, 1]
            L[j, k] = (A[j, k] - sum_for_crout(L, U, k, j, k)) / U[k, k]
            # L[j, k] /= U[k, k]

    return L, U


def getLeadColIdx(A, r):
    _, nc = A.shape
    for i in range(nc):
        if abs(A[r, i]) > 0:
            return i

    return 999999999999


def pivot(A, startRow):
    nr, nc = A.shape
    leadr, leadc = startRow, getLeadColIdx(A, startRow)
    for r in range(startRow + 1, nr):
        lead = getLeadColIdx(A, r)
        if (lead < leadc):
            leadr, leadc = r, lead
    return leadr, leadc


def RREF(A):
    nr, nc = A.shape
    pr, pc = pivot(A, 0)
    for r in range(nr - 1):

        # if pivot is not element of vector b
        if (pc < nc - 1):
            # if pivot's row is not where it suppossed to be
            if (pr != r):
                A[[pr, r]] = A[[r, pr]]  # swap it
            # if pivot element is not 1
            if A[r][pc] != 1:
                # convert it to 1
                mul = 1 / A[r, pc]
                for c in range(nc):
                    A[r, c] *= mul

            # make zeros for all rows beside pr in the same column
            for x in range(r, nr):
                # print(x)
                # print(A)
                if (r != x):
                    if (A[x, pc] != 0):
                        mul = A[x, pc] / A[r, pc]
                        for c in range(nc):
                            A[x, c] -= A[r, c] * mul
                            # print(A)

        elif pc == nc - 1:
            break
        pr, pc = pivot(A, r + 1)

    # last iter
    if (pc < nc - 1 and A[nr - 1, pc] != 1):
        if A[nr - 1, pc] != 1:
            # convert it to 1
            mul = 1 / A[nr - 1, pc]
            for c in range(nc):
                A[nr - 1, c] *= mul

        # for x in range(nr - 1):
        #     # print(x)
        #     # print(A)
        #     if (A[x, pc] != 0):
        #         mul = A[x, pc] / A[nr-1, pc]
        #         for c in range(nc):
        #             A[x, c] -= A[nr-1, c] * mul

    return A


def eigen_vector(eig, cov):
    n = cov.shape[0]
    eig_v = [[0 for i in range(len(cov))] for j in range(len(cov))]
    b = np.zeros([n, 1])
    for val in eig:
        copy = np.copy(cov)
        for i in range(0, len(copy)):
            copy[i, i] = val - copy[i, i]
        aug = np.concatenate((copy, b), axis=1)
        print(aug)
        rref = RREF(aug)
        print(rref)


# A = np.matrix('1 2; 3 4')
# print(eigen_value(A))
# print(np.linalg.eig(A))
# A = np.array([[1, -1, 0, 0], [-1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.double)
# RREF(A)
# print(A)
# print(L)
# print(U)
# print(np.dot(L, U))

# y = forward_sub_zeros(L)
# print(y)
