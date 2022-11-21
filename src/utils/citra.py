import cv2 as cv
import numpy as np


def resize(img):
    ''' CROP AND RESIZE IMAGE AND CONVERTING TO GRAYSCALE IMAGE '''
    # img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # height = len(img_gray)
    # width = len(img_gray[0])

    # if (height > width):
    #     crop_img = img_gray[int(height/2-width/2):int(height/2+width/2), 0:width]
    # else:
    #     crop_img = img_gray[0:height, int(
    #         width/2-height/2):int(width/2+height/2)]

    resized_img = cv.resize(img, (256, 256))
    return resized_img


def detect_crop_face(img):
    ''' DETECT FACE AND CROP IMAGE '''
    # Convert the image to gray
    gray_image = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Make a copy of the original image to draw face detections on
    image_copy = np.copy(img)

    # Detect faces in the image using pre-trained face dectector
    face_cascade = cv.CascadeClassifier(
        cv.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(gray_image, 1.25, 6)

    # Print number of faces found
    # print('Number of faces detected:', len(faces))

    face_crop = []
    for f in faces:
        x, y, w, h = [v for v in f]
        cv.rectangle(image_copy, (x, y), (x+w, y+h), (255, 0, 0), 3)
        # Define the region of interest in the image
        face_crop.append(gray_image[y:y+h, x:x+w])

    return len(faces), face_crop


# img = cv.imread(r'full_test/Alexandra Daddario4_377.jpg')
# face = detect_crop_face(img)
# cv.imshow('face', face)
# cv.waitKey(0)
