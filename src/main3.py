''' FACE RECOGNITION WITHOUT FEATURE EXTRACTION '''

import cv2 as cv
import numpy as np
import os
from math import sqrt
import utils.euclidean_algorithm as eucl
import utils.citra as ctr
import utils.eigen_value as eig

''' PENGOLAHAN CITRA '''


def extract_image(image_path):
    ''' EXTRACT IMAGE USING OPEN CV ONLY (WITHOUT FEATURE EXTRACTION) '''
    image = cv.imread(image_path)
    image = ctr.resize(image)
    # normalizedImg = np.zeros((255, 255))
    # normalizedImg = cv.normalize(image,  normalizedImg, 0, 255, cv.NORM_MINMAX)
    # print(normalizedImg)
    # cv.imshow('dst_rt', normalizedImg)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # print(image)
    # image = image.flatten().reshape((256, 256))
    # cv.imshow('displaymywindows', image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return (image.flatten())


def batch_extractor_2(images_path):
    ''' BATCH EXTRACTOR USING ONLY OPEN CV (WITHOUR FEATURE EXTRACTION) '''
    files = [os.path.join(images_path, p)
             for p in sorted(os.listdir(images_path))]

    result = []
    res_name = []
    sum = [0 for i in range(256 * 256)]
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        extract = extract_image(f)
        result.append(extract)
        sum = np.add(sum, extract)
        res_name.append(name)

    # saving all our feature vectors in pickled file
    # with open(pickled_db_path, 'w') as fp:
    #     pickle.dump(result, fp)
    print(result)
    mean = np.divide(sum, len(result))
    print(mean)
    return np.transpose(result), res_name, mean  # 2048 x m


''' TESTING PURPOSES '''


def test_batch(mean, ef, y, res_name):
    ''' BATCH TESTING, INPUT IS A FOLDER
        USED TO TEST THE AMOUNT OF FA AND FR FOR PICTURES OF THE SAME PERSON
    '''
    folder = input("INPUT FOLDER (with relative path): ")
    # OPTION: batch_extractor and batch_extractor_2
    result, res_name_test, _ = batch_extractor_2(folder)
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


def test_image(mean, ef, y, res_name):
    ''' TEST ONE PICTURE '''
    f = input('FILE NAME (relative to test/foto): ')
    ex = np.transpose(extract_image('test/foto/' + f + ".jpg"))
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


def menu():
    # extract_image(r"test/foto_testing/Chris Pratt1_723.jpg")
    folder = input("FOLDER NAME: ")
    # OPTION: batch_extractor and batch_extractor
    result, res_name, mean = batch_extractor_2(folder)
    print("RESULT")
    print(result.shape)
    # result = result.reshape((256, 256))
    print("\n\n")
    # mean = eucl.mean_mat(result)
    print("MEAN MATRIX")
    print(mean)
    print(mean.shape)

    ''' PRINT MEAN FACE '''
    meanFace = mean.reshape((256, 256))
    meanFace = np.array(meanFace, dtype=np.uint8)
    print(meanFace)
    cv.imshow('displaymywindows', meanFace)
    cv.waitKey(0)
    cv.destroyAllWindows()
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
    print(e)

    y = eig.proj(e, A)
    # print("EIGEN FACE")
    # print(y)
    # print("\n\n")
    while (True):
        test_batch(mean, e, y, res_name)
    # while (True):
    #     test_image(mean, e, y, res_name)


menu()
