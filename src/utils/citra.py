import cv2 as cv


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
