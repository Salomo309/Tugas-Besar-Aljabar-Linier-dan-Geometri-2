import main2
import cv2 as cv
import numpy as np
import os
from math import sqrt
import utils.euclidean_algorithm as eucl
import utils.citra as ctr
import utils.eigen_value as eig
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from datetime import datetime
import time
import webcam


window = Tk()

window.geometry("1200x700")
window.configure(bg="#eccdff")
global foldername
global result, res_name, mean, e, y
fileChosen = False
folderChosen = False

foldername = ''
filename = ''

seconds = 0
ms = 0


def getTime(start):
    stop = datetime.now().timestamp()
    duration = round(stop-start, 2)
    print("duration =", duration)
    canvas.itemconfig(
        time_label,
        text=duration
    )

    canvas.itemconfig(
        process,
        text=""
    )


# def update_stopwatch():
#     global seconds
#     global ms
#     if ms < 59:
#         ms += 1
#     elif ms == 59:
#         ms = 0
#         seconds += 1
#     # Update Label.
#     time_string = "{:02d}:{:02d}".format(seconds, ms)
#     canvas.itemconfig(
#         time_label,
#         text=time_string
#     )
#     window.after(10, update_stopwatch)  # Call again in 10 millisecs.

def changeProcessText():
    global filename, foldername

    if (foldername != '' and filename != ''):
        canvas.itemconfig(
            process,
            text="Processing..."
        )

        startProcess()


def startProcess():
    global mean, e, y, res_name
    global filename, foldername

    if (foldername != '' and filename != ''):
        start = datetime.now().timestamp()
        print('Start process...')

        # update_stopwatch()
        result, res_name = main2.batch_extractor(foldername)
        print("RESULT")
        print(result.shape)
        # result = result.reshape((256, 256))
        print("\n\n")
        mean = eucl.mean_mat(result)
        print("MEAN MATRIX")
        print(mean)
        print(mean.shape)

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

        path = main2.test_image(mean, e, y, res_name, window.filename)

        openClosestResult(path)
        getTime(start)


def openFolder():
    global myFolder
    global folderChosen
    global foldername
    global result, res_name, mean, e, y
    window.foldername = filedialog.askdirectory()
    foldername = window.foldername.split(
        '/')[len(window.foldername.split('/'))-1]

    if (foldername != ''):
        canvas.itemconfig(
            cfo,
            text=foldername
        )
    # else:
    #     canvas.itemconfig(
    #         cfo,
    #         text="No Folder Chosen"
    #     )


def openFile():
    global myImage
    global fileChosen
    global mean, e, y, res_name
    global filename
    window.filename = filedialog.askopenfilename(
        initialdir="ALGEO02-21063", title="Select an image", filetypes=(("JPG Files", "*.jpg"), ("PNG Files", "*.png"), ("All Files", "*.*")))

    filename = window.filename.split('/')[len(window.filename.split('/'))-1]
    if (filename != ''):
        canvas.itemconfig(
            cf,
            text=filename
        )

    # path = main2.test_image(mean, e, y, res_name, filename)

    img = Image.open(window.filename)
    myImage = ImageTk.PhotoImage(img)
    new_width = 256
    new_height = int(new_width * myImage.height() / myImage.width())

    if (new_height > 275):
        new_height = 275
        new_width = int(new_width * myImage.width() / myImage.height())

    resized = img.resize((new_width, new_height), Image.ANTIALIAS)
    myImage = ImageTk.PhotoImage(resized)

    testImage = canvas.create_image(
        675, 360,
        image=myImage)

    # openClosestResult(path)


def openCamera():
    global myImage
    global filename
    webcam.captureWebcam()

    filename = 'image_from_webcam.jpg'
    img = Image.open(f'../test/foto/{filename}')
    myImage = ImageTk.PhotoImage(img)

    new_width = 256
    new_height = int(new_width * myImage.height() / myImage.width())

    if (new_height > 275):
        new_height = 275
        new_width = int(new_width * myImage.width() / myImage.height())

    resized = img.resize((new_width, new_height), Image.ANTIALIAS)
    myImage = ImageTk.PhotoImage(resized)

    if (filename != ''):
        canvas.itemconfig(
            cf,
            text=filename
        )

    testImage = canvas.create_image(
        675, 360,
        image=myImage)


def openClosestResult(path):
    global closestRes
    global foldername

    img = Image.open(window.foldername + f'/{path}')
    closestRes = ImageTk.PhotoImage(img)

    new_width = 256
    new_height = int(new_width * closestRes.height() / closestRes.width())

    if (new_height > 275):
        new_height = 275
        new_width = int(new_width * closestRes.width() / closestRes.height())

    resized = img.resize((new_width, new_height), Image.ANTIALIAS)
    closestRes = ImageTk.PhotoImage(resized)

    res = canvas.create_image(
        995, 360,
        image=closestRes)

    if (path != ''):
        canvas.itemconfig(
            result,
            text=path
        )


canvas = Canvas(
    window,
    bg="#eccdff",
    height=700,
    width=1200,
    bd=0,
    highlightthickness=0,
    relief="ridge")
canvas.place(x=0, y=0)


img0 = PhotoImage(file=f"GUI/img0.png")
b0 = Button(
    image=img0,
    borderwidth=0,
    highlightthickness=0,
    command=openFolder,
    relief="flat")
b0.place(
    x=85, y=233,
    width=149,
    height=41)


img1 = PhotoImage(file=f"GUI/img1.png")
b1 = Button(
    image=img1,
    borderwidth=0,
    highlightthickness=0,
    command=openFile,
    relief="flat")
b1.place(
    x=85, y=428,
    width=149,
    height=41)


img3 = PhotoImage(file=f"GUI/img3.png")
b3 = Button(
    image=img3,
    borderwidth=0,
    highlightthickness=0,
    command=changeProcessText,
    relief="flat")
b3.place(
    x=543, y=560,
    width=149,
    height=41)


img2 = PhotoImage(file=f"GUI/img2.png")
b2 = Button(
    image=img2,
    borderwidth=0,
    highlightthickness=0,
    command=openCamera,
    relief="flat")
b2.place(
    x=347, y=385,
    width=106,
    height=29)


result = canvas.create_text(
    115, 620.5,
    text="None",
    fill="#540097",
    font=("Poppins-Regular", int(16)))

process = canvas.create_text(
    770, 580,
    text="",
    fill="#540097",
    font=("Poppins-Regular", int(16)))

time_label = canvas.create_text(
    739.5, 627.5,
    text="00.00",
    fill="#540097",
    font=("Poppins-Regular", int(16)))

cf = canvas.create_text(
    160.0, 493.0,
    text="No File Chosen",
    fill="#540097",
    font=("Poppins-Regular", int(16)))

cfo = canvas.create_text(
    175, 298.0,
    text="No Folder Chosen",
    fill="#540097",
    font=("Poppins-Regular", int(16)))

background_img = PhotoImage(file=f"GUI/background.png")
background = canvas.create_image(
    599.5, 343.0,
    image=background_img)

# window.resizable(False, False)
window.title("Face Recognition")
window.mainloop()
