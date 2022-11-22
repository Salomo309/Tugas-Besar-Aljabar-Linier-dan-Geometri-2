import main
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
THRESHOLD = 0.997533

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
        result, res_name = main.batch_extractor(foldername)
        mean = eucl.mean_mat(result)
        A = eucl.sub_mat(result, mean)

        C1 = eucl.covariant(A)

        k = min(10, len(C1) - 1)
        evals, eigh = eig.qr_iteration(C1, k)

        e = eig.eigen_vector(A, eigh)
        y = eig.proj(e, A)
        path, minED = main.test_image(mean, e, y, res_name, window.filename)
        openClosestResult(path, minED)
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


def openCamera():
    global myImage
    global filename
    webcam.captureWebcam()

    window.filename = '../test/foto/image_from_webcam.jpg'
    filename = 'image_from_webcam.jpg'
    img = Image.open(f'{window.filename}')
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


def openClosestResult(path, minED):
    global closestRes
    global foldername
    global THRESHOLD

    img = Image.open(window.foldername + f'/{path}')
    closestRes = ImageTk.PhotoImage(img)

    new_width = 256
    new_height = int(new_width * closestRes.height() / closestRes.width())

    if (new_height > 275):
        new_height = 275
        new_width = int(new_width * closestRes.width() /
                        closestRes.height())

    resized = img.resize((new_width, new_height), Image.ANTIALIAS)
    closestRes = ImageTk.PhotoImage(resized)

    if (path != '' and minED <= THRESHOLD):
        closestRes = ImageTk.PhotoImage(resized)
        textResult = minED
    else:
        print('NO MATCH')
        closestRes = ImageTk.PhotoImage(file=f"GUI/placeholder.jpg")
        textResult = 'None'

    canvas.itemconfig(
        result,
        text=textResult
    )
    res = canvas.create_image(
        995, 360,
        image=closestRes
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
    215, 620.5,
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
