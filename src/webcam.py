import cv2
import os


def captureWebcam():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Webcam")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Webcam", frame)

        k = cv2.waitKey(1)
        if k % 256 == 32:
            # SPACE pressed
            img_name = f"image_from_webcam.jpg"
            cv2.imwrite(os.path.join('../test/foto/', img_name), frame)
            break

    cam.release()
    cv2.destroyAllWindows()
