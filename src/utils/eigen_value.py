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
