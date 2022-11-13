import cv2 as cv
import numpy as np
import EuclideanAlgorithm as eucl
import EigenValue as ev

# testing pake 2 foto
img1 = cv.imread(r'test/foto_testing/Adriana Lima0_0.jpg')
img2 = cv.imread(r'test/foto_testing/Jimmy Fallon0_2757.jpg')

temp = [img1, img2]

arr = []

for img in temp:
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height = len(img_gray)
    width = len(img_gray[0])

    if (height > width):
        crop_img = img_gray[int(height/2-width/2):int(height/2+width/2), 0:width]
    else:
        crop_img = img_gray[0:height, int(
            width/2-height/2):int(width/2+height/2)]

    resized_img = cv.resize(crop_img, (5, 5))
    arr.append(resized_img)
    # print(resized_img)
    # print()

    # cv.imshow('Resize image', resized_img)
    # cv.waitKey(0)


meanmat = eucl.mean(arr)

print("matriks nilai tengah training image")
for row in meanmat:
    for col in row:
        print(col, end=' ')
    print()

print()

for mat in arr:
    for i in range(5):
        for j in range(5):
            mat[i][j] -= meanmat[i][j]

print("selisih training image dengan nilai tengah")
for mat in arr:
    print("matriks")
    for row in mat:
        for col in row:
            print(col, end=' ')
        print()

print()

new_arr = eucl.concat(arr)
print("concat matriks selisih")
print(new_arr)

print()

cov = eucl.covarian(new_arr)
print("matriks kovarian")
print(cov)

print("eigen value with qr decomposition")
print(ev.eigen_value(cov))

print("eigen value with qr decomposition with shift")
eig_val = ev.eigen_value(cov)
print(eig_val)

print("eigen value from lib numpy")
print(np.sort(np.linalg.eigvals(cov)))
v, w = np.linalg.eig(cov)
print(v)
print(w)

print("eigen vector")
# cov2 = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 3]])
# eig_val = ev.eigen_value_with_shift(cov2)
# print(eig_val.astype(int))
print(ev.eigen_vector(eig_val, cov))

# a = [
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9],
# ]
# b = [
#     [11, 12, 13],
#     [14, 15, 16],
#     [17, 18, 19],
# ]
# x = np.concatenate((a, b), axis=1)
# print(x)
