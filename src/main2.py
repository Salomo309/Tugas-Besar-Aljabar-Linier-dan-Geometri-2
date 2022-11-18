import cv2 as cv
import numpy as np
import os
from scipy.linalg import null_space
from math import sqrt

''' PENGOLAHAN CITRA '''


def extract_image(image_path):
    ''' EXTRACT IMAGE USING OPEN CV ONLY (WITHOUT FEATURE EXTRACTION) '''
    image = cv.imread(image_path)
    image = resize(image)
    return image.flatten()


def batch_extractor_2(images_path):
    ''' BATCH EXTRACTOR USING ONLY OPEN CV (WITHOUR FEATURE EXTRACTION) '''
    files = [os.path.join(images_path, p)
             for p in sorted(os.listdir(images_path))]

    result = []
    res_name = []
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result.append(extract_image(f))
        res_name.append(name)

    # saving all our feature vectors in pickled file
    # with open(pickled_db_path, 'w') as fp:
    #     pickle.dump(result, fp)
    # print(result)
    return np.transpose(result), res_name  # 2048 x m


def resize(img):
    ''' CROP AND RESIZE IMAGE AND CONVERTING TO GRAYSCALE IMAGE '''
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height = len(img_gray)
    width = len(img_gray[0])

    if (height > width):
        crop_img = img_gray[int(height/2-width/2):int(height/2+width/2), 0:width]
    else:
        crop_img = img_gray[0:height, int(
            width/2-height/2):int(width/2+height/2)]

    resized_img = cv.resize(crop_img, (256, 256))
    return resized_img


def extract_features(image_path, vector_size=32):
    ''' EXTRACT IMAGE FEATURE '''
    ''' SOURCE: https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774 '''
    image = cv.imread(image_path)
    image = resize(image)
    try:

        # image = cv.resize(image, (256, 256))
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them.
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv.error as e:
        print('Error: ', e)
        return None

    return dsc


def batch_extractor(images_path):
    ''' EXTRACT IMAGE BATCH 
        SOURCE: https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774
    '''
    files = [os.path.join(images_path, p)
             for p in sorted(os.listdir(images_path))]
    print("Extracting", len(files), "Files...")
    result = []
    res_name = []
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result.append(extract_features(f))
        res_name.append(name)
    # TODO: SAVE EXTRACT RESULT IN FILE
    # saving all our feature vectors in pickled file
    # with open(pickled_db_path, 'w') as fp:
    #     pickle.dump(result, fp)
    # print(result)
    return np.transpose(result), res_name  # 2048 x m


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

    A = np.array(arr)
    print(A.shape)
    for i in range(n):
        A[:, i] = np.subtract(A[:, i], mean)

    return A  # 2048 x m


def covariant(A):
    ''' COVARIANT OF MATRIX A (2048 X M) '''
    return np.matmul(np.transpose(A), A)  # m x m


def qr(A):
    ''' QR DECOMPOSITION USING SCHWARZ-RUTISHAUSER ALGORITHM
        SOURCE: https://towardsdatascience.com/can-qr-decomposition-be-actually-faster-schwarz-rutishauser-algorithm-a32c0cde8b9b
    '''
    # get shape
    A = np.array(A)
    _, n = np.shape(A)

    Q = np.array(A)
    R = np.zeros((n, n))

    for k in range(n):
        for i in range(k):
            R[i, k] = np.transpose(Q[:, i]).dot(Q[:, k])
            Q[:, k] = Q[:, k] - (R[i, k] * Q[:, i])

        R[k, k] = np.linalg.norm(Q[:, k])
        Q[:, k] = Q[:, k] / R[k, k]

    return -Q, -R


def qr_iteration(C1, k):
    ''' SIMULTANEOUS QR ITERATION FOR EIGENVALUE R AND EIGENVECTOR Q (FROM REDUCED COVARIANCE MATRIX)
        k is the amount of eigenvector we want (the best eigenvalues)
        usually k = 0.1 M because 90% of the total variance is contained in the first 5% to 10% eigenvectors
        SOURCE: https://www.researchgate.net/publication/260880521 
    '''
    n, m = C1.shape
    Q = np.random.rand(n, k)
    Q, _ = qr(Q)

    for i in range(2000):
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
        E[:, i] = normalize(A.dot(v[:, i]))

    return E  # 2048 x d, where d is the desired amount of eigen vectors we want to find


def normalize(E):
    '''
        RETURN NORMALIZED OF VECTOR E
    '''
    E = E / np.linalg.norm(E)

    return E  # 2048 x d


def proj(E, A):
    '''
    PROJECTION OF DATA MATRIX A AND CALCULATION OF y_i VECTORS OF MATRIX Y =(y1,...,yM )
    OR EIGEN FACE
    '''
    return np.transpose(E).dot(A)  # d x m


def euc_distance(a, b):
    '''
        RETURN EUCLEDIAN DISTANCE OF TWO VECTOR
    '''
    return np.linalg.norm(a - b)


''' TESTING PURPOSES '''


def test_batch(mean, ef, y, res_name):
    ''' BATCH TESTING, INPUT IS A FOLDER 
        USED TO TEST THE AMOUNT OF FA AND FR FOR PICTURES OF THE SAME PERSON
    '''
    folder = input("INPUT FOLDER (with relative path): ")
    # OPTION: batch_extractor and batch_extractor_2
    result, res_name_test = batch_extractor_2(folder)
    for j in range(len(result[0])):
        print("\nTEST:", res_name_test[j])
        sub = result[:, j] - mean
        omega = np.transpose(ef).dot(sub)  # d x 1
        min = 999999999
        max = -1
        min_id = -1
        # max_id = -1
        for i in range(y.shape[1]):
            # print("EUC DISTANCE", res_name[i])
            ed = euc_distance(omega, y[:, i])
            # print(ed)
            if (ed < min):
                min = ed
                min_id = i
            if (ed > max):
                max = ed
                # max_id = i

        print("RESULT:", res_name[min_id])
        print("DISTANCE MIN:", min)
        print("DISTANCE MAX:", max)
        # TODO: CALCULATE THRESHOLD
        print("THRESHOLD:", sqrt(max / 2))


def test_image(mean, ef, y, res_name):
    ''' TEST ONE PICTURE '''
    f = input('FILE NAME (relative to test/foto): ')
    ex = np.transpose(extract_features('test/foto/' + f + ".jpg"))
    sub = ex - mean
    omega = np.transpose(ef).dot(sub)  # d x 1
    min = 999999999
    max = -1
    min_id = -1
    max_id = -1
    for i in range(y.shape[1]):
        print("EUC DISTANCE", res_name[i])
        ed = euc_distance(omega, y[:, i])
        print(ed)
        if (ed < min):
            min = ed
            min_id = i
        if (ed > max):
            max = ed
            max_id = i

    print("\nRESULT:", res_name[min_id])


def menu():
    folder = input("FOLDER NAME: ")
    # OPTION: batch_extractor and batch_extractor_2
    result, res_name = batch_extractor_2(folder)
    print("RESULT")
    print(result.shape)

    print("\n\n")
    mean = mean_mat(result)
    print("MEAN MATRIX")
    print(mean)
    print(mean.shape)

    print("\n\n")

    A = np.zeros(result.shape)
    for i in range(result.shape[1]):
        A[:, i] = result[:, i] - mean

    print("TRAINING MATRIX (2048XM)")
    print(A)
    print(A.shape)

    print("\n\n")

    C1 = covariant(A)
    print("COVARIANT MXM")
    print(C1)

    print("\n\n")

    evals, eigh = qr_iteration(C1, len(C1) // 10)
    # evals = eigen_value(C1)
    # eigh = eigen_vector(A, evals, C1)
    # v, w = np.linalg.eig(C1)
    # print("OWNED LIB vs NUMPY")
    # print(v)
    # print(evals)
    # print("===========")
    # print("\n\n")
    # print(eigh[1])
    # print("============")
    # np.set_printoptions(threshold=sys.maxsize)

    # print(w[1])
    e = eigen_vector(A, eigh)
    y = proj(e, A)
    print("EIGEN FACE")
    # print(y)
    # print("\n\n")
    while (True):
        test_batch(mean, e, y, res_name)
    # while (True):
    #     test_image(mean, e, y, res_name)


menu()


# TPK: TEMPAT PEMBUANGAN KODE
# def extract_image(image_path):
#     image = cv.imread(image_path)
#     image = resize(image)
#     return image.flatten()


# def batch_extractor_2(images_path):
#     files = [os.path.join(images_path, p)
#              for p in sorted(os.listdir(images_path))]

#     result = []
#     res_name = []
#     for f in files:
#         print('Extracting features from image %s' % f)
#         name = f.split('/')[-1].lower()
#         result.append(extract_image(f))
#         res_name.append(name)

#     # saving all our feature vectors in pickled file
#     # with open(pickled_db_path, 'w') as fp:
#     #     pickle.dump(result, fp)
#     # print(result)
#     return np.transpose(result), res_name  # 2048 x m


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
#     for i in range(len(eig)):
#         if (abs(eig[i]) < 0.0000001):
#             continue
#         copy = np.array(cov, dtype=np.double)
#         tes = np.subtract(np.multiply(I, eig[i]), copy)
#         # aug = np.concatenate((copy, b), axis=1)
#         # aug = Matrix(aug)
#         # print(aug)
#         print(null_space(tes))
#         v_i = np.transpose(null_space(tes))
#         # print("null")
#         if (len(v_i) == 0):
#             v_i = [0 for j in range(n)]
#         else:
#             v_i = v_i[0]
#             eig_v.append(v_i)
#     #     u_i = np.matmul(A, v_i)
#     #     # print(u_i)
#     #     eig_vec.append(u_i)
#     # print(eig_v)
#     return np.transpose(np.array(eig_v, dtype=np.double))
