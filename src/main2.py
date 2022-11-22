''' FACE RECOGNITION WITH FEATURE EXTRACTION '''

import cv2 as cv
import numpy as np
import os
from math import sqrt
import utils.euclidean_algorithm as eucl
import utils.citra as ctr
import utils.eigen_value as eig

''' PENGOLAHAN CITRA '''


def extract_features(image_path, vector_size=32):
    ''' EXTRACT IMAGE FEATURE '''
    ''' SOURCE: https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774 '''
    image = cv.imread(image_path)
    length, images = ctr.detect_crop_face(image)
    if length == 0:
        print("No face detected:", image_path)
        return []

    try:
        image = images[0]
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
        needed_size = (vector_size * 256)
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
    images_path = f"../test/{images_path}"
    files = [os.path.join(images_path, p)
             for p in sorted(os.listdir(images_path))]
    print("Extracting", len(files), "Files...")
    result = []
    res_name = []
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        extracted = extract_features(f)
        if len(extracted) > 0:
            result.append(extract_features(f))
            res_name.append(name)
    # TODO: SAVE EXTRACT RESULT IN FILE
    # saving all our feature vectors in pickled file
    # with open(pickled_db_path, 'w') as fp:
    #     pickle.dump(result, fp)
    # print(result)
    return np.transpose(result), res_name  # 2048 x m


''' TESTING PURPOSES '''


def test_batch(mean, ef, y, res_name):
    ''' BATCH TESTING, INPUT IS A FOLDER
        USED TO TEST THE AMOUNT OF FA AND FR FOR PICTURES OF THE SAME PERSON
    '''
    folder = input("INPUT FOLDER (with relative path): ")
    # OPTION: batch_extractor and batch_extractor_2
    result, res_name_test = batch_extractor(folder)
    count = 0
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
            ed = eucl.euc_distance(omega, y[:, i])
            # print(ed)
            if (ed < min):
                min = ed
                min_id = i
            if (ed > max):
                max = ed
                # max_id = i
        person_test = "".join(filter(lambda x: x.isalpha(), res_name_test[j]))
        person_result = "".join(
            filter(lambda x: x.isalpha(), res_name[min_id]))
        if (person_result == person_test):
            count += 1
        print("RESULT:", res_name[min_id])
        print("DISTANCE MIN:", min)
        print("DISTANCE MAX:", max)
        # TODO: CALCULATE THRESHOLD
        print("THRESHOLD:", sqrt(max / 2))
    print("TEST CONCLUDED. ACCURACY:", round(
        100 * count / len(result[0]), 2), "%")


def test_image(mean, ef, y, res_name, f):
    ''' TEST ONE PICTURE '''
    # f = input('FILE NAME (relative to test/foto): ')
    ex = np.transpose(extract_features(f))
    sub = ex - mean
    omega = (np.transpose(ef).dot(sub))  # d x 1
    min = 999999999
    max = -1
    min_id = -1
    max_id = -1
    for i in range(y.shape[1]):
        print("EUC DISTANCE", res_name[i])
        ed = eucl.euc_distance(omega, y[:, i])
        print(ed)
        if (ed < min):
            min = ed
            min_id = i
        if (ed > max):
            max = ed
            max_id = i

    print("\nRESULT:", res_name[min_id])
    return res_name[min_id].split('\\')[1], min


def menu():
    # extract_image(r"test/foto_testing/Chris Pratt1_723.jpg")
    folder = input("FOLDER NAME: ")
    # OPTION: batch_extractor and batch_extractor
    result, res_name = batch_extractor(folder)
    print("RESULT")
    print(result.shape)
    # result = result.reshape((256, 256))
    print("\n\n")
    mean = eucl.mean_mat(result)
    print("MEAN MATRIX")
    print(mean)
    print(mean.shape)
    # meanFace = mean.reshape((256, 256))
    # meanFace = np.array(meanFace, dtype=np.uint8)
    # print(meanFace)
    # cv.imshow('displaymywindows', meanFace)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # normalizedImg = np.zeros((255, 255))
    # normalizedImg = cv.normalize(
    #     eigenFace,  normalizedImg, 0, 255, cv.NORM_MINMAX)

    print("\n\n")

    print("TRAINING MATRIX (2048XM)")
    A = eucl.sub_mat(result, mean)
    print(A)
    print(A.shape)

    print("\n\n")

    C1 = eucl.covariant(A)
    print("COVARIANT MXM")
    print(C1)

    print("\n\n")

    evals, eigh = eig.qr_iteration(C1, min(10, len(C1) // 10))
    # evals = eigen_value(C1)
    # eigh = eigen_vector(A, evals, C1)
    v, w = np.linalg.eig(C1)
    # print("OWNED LIB vs NUMPY")
    print(v)
    print(evals)
    print("===========")
    print("\n\n")
    print(eigh[1])
    print("============")
    # np.set_printoptions(threshold=sys.maxsize)
    print(w[1])
    e = eig.eigen_vector(A, eigh)
    # print(e)

    y = eig.proj(e, A)
    # print("EIGEN FACE")
    # print(y)
    # print("\n\n")
    # while (True):
    #     test_batch(mean, e, y, res_name)
    while (True):
        test_image(mean, e, y, res_name, "tes")


# menu()


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
