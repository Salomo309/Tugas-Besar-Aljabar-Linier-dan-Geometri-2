import cv2
import numpy as np
import scipy
from matplotlib.pyplot import imread
from sympy import Matrix
import random
import os
import matplotlib.pyplot as plt
import EuclideanAlgorithm as eucl
import EigenValue as ev

# Feature extractor


def extract_features(image_path, vector_size=32):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        # height = len(img_gray)
        # width = len(img_gray[0])

        # if (height > width):
        #     crop_img = img_gray[int(height/2-width/2):int(height/2+width/2), 0:width]
        # else:
        #     crop_img = img_gray[0:height, int(
        #         width/2-height/2):int(width/2+height/2)]

        # image = cv2.resize(image, (256, 256))
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
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
    except cv2.error as e:
        print('Error: ', e)
        return None

    return dsc


def mean_mat(arr):
    n = arr.shape[1]

    sum_mat = np.array(arr[:, 0])
    for i in range(1, n):
        sum_mat += arr[:, i]

    return sum_mat / n


def sub_mat(arr, mean):
    n = arr.shape[1]

    copy = np.array(arr)
    print(copy.shape)
    for i in range(n):
        copy[:, i] = np.subtract(copy[:, i], mean)
    return copy


def covariant(A):

    return np.matmul(np.transpose(A), A)


def batch_extractor(images_path):
    files = [os.path.join(images_path, p)
             for p in sorted(os.listdir(images_path))]

    result = []
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result.append(extract_features(f))

    # saving all our feature vectors in pickled file
    # with open(pickled_db_path, 'w') as fp:
    #     pickle.dump(result, fp)

    return result


result = batch_extractor("test/foto_testing/")
print(result)

print(len(result[0]))
result = np.transpose(result)
print(result)
mean = mean_mat(result)
# print(result)
print("\n\nMEAN")
print(mean)
print(mean.shape)
print("\n\nSELISIH MATRIKS DAN KONKATENASI")
concat_sub = sub_mat(result, mean)
print(concat_sub)
print(concat_sub.shape)

cov = covariant(concat_sub)
# print(mean)
print("\n\nCOVARIANT")
print(cov)
print(cov.shape)

eig = ev.eigen_value(cov)
# print("check cov")
# print(cov)
cov = np.array(cov, dtype=np.double)
print("\n\nEIGEN VALUE")
print(eig)
print(eig.shape)

# print("check cov")
# print(cov)
# print(np.sort(np.linalg.eig(cov)))
print("\n\nEIGEN VECTOR")
eig_vec = ev.eigen_vector(concat_sub, eig, cov)
print(eig_vec)
print(eig_vec.shape)

v, w = np.linalg.eig(cov)
print("\n\nEIGEN VALUE FROM NUMPY")
print(v)
# print("\n\nEIGEN VECTOR FROM NUMPY")
# print(w)

print("\n\nEIGEN FACE PERSON 1")
print("\nSUB MAT")
print(concat_sub[8])
print(concat_sub.shape)


f = extract_features("test/foto_testing/Bill Gates5_583.jpg")
print("\n\nTEST FACE")
print(f)
sub = np.subtract(f, mean)
print("\n\nTEST FACE NORM")
print(sub)
ef_t = ev.eigen_face(eig_vec, sub)
print("\n\nTEST FACE EIGEN FACE")
print(ef_t)


for i in range(result.shape[1]):
    if (i == 2):
        print('\n')
        print(f)
        print(result[i])
    ef = ev.eigen_face(eig_vec, concat_sub[:, i])
    # print("\n\nEIG FACE")
    # print(ef)
    # print("\n\nSELISIH TEST FACE DAN EIGEN FACE KE-", i)
    selisih = ef - ef_t
    # print(selisih)
    print('\n\nEUC DISTANCE', i)
    distance = np.linalg.norm(selisih)
    print(distance)

# ef_1 = ev.eigen_face(eig_vec, concat_sub[8])
# print("\n\nEIG FACE")
# print(ef_1)

# ef_2 = ev.eigen_face(eig_vec, concat_sub[0])
# print("\n\nEIG FACE")
# print(ef_2)
# print("\n\nSELISIH TEST FACE DAN EIGEN FACE ORANG YANG SAMA")
# selisih = ef_1 - ef_t
# print(selisih)

# print("\n\nSELISIH TEST FACE DAN EIGEN FACE ORANG YANG BEDA")
# selisih1 = ef_2 - ef_t
# print(selisih1)

# print('\n\nEUC DISTANCE 1')
# distance = np.linalg.norm(selisih)
# print(distance)

# print('\n\nEUC DISTANCE 2')
# distance1 = np.linalg.norm(selisih1)
# print(distance1)
