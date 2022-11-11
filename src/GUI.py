from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog
import sqlite3

root = Tk()
root.title("GUI")
root.geometry("1530x780")
root["bg"] = "#FFDDD2"

# Database

# Create a database or connect to one
# conn = sqlite3.connect()

title = Label(root, text="Face Recognition Salomo Ganteng", justify='center', font=("Arial",20), border=50, bg="#FFDDD2")
title.grid(row=0,column=0, columnspan=3)
# line = Frame(root, width=1410, height=1,bg="black", border=100).pack()

# Insert Frame
insertFrame1 = Frame(root,width=510,height=610, bg="#FFB9B9")
insertFrame1.grid(row=1,column=0, rowspan=4)

def openFolder():
    global myFolder
    root.foldername = filedialog.askdirectory()
    foldername = root.foldername.split('/')[len(root.foldername.split('/'))-1]
    folder = Label(root, text=foldername, font=(16)).grid(row=2, column=0)
    

folderButton = Button(root, bg="white", text=" Please Insert a Folder", command=openFolder, justify="left", font=(30))
folderButton.grid(row=1, column=0)

def openFile():
    global myImage
    root.filename = filedialog.askopenfilename(initialdir="ALGEO02-21063", title="Select an image", filetypes=(("JPG files", "*.jpg"),("All files","*.*")))
    filename = root.filename.split('/')[len(root.filename.split('/'))-1]
    file = Label(root, text=filename, font=(16)).grid(row=4, column=0)
    myImage = ImageTk.PhotoImage(Image.open(root.filename))
    
    insertImage = Label(image=myImage).grid(row=1,column=1, rowspan=4)

imageButton = Button(root, bg="white", text="Please Insert an Image", command=openFile, justify="left", font=(30))
imageButton.grid(row=3, column=0)

# Test Image Frame
insertFrame2 = Frame(root,width=510,height=610, bg="#FFB9B9")
insertFrame2.grid(row=1,column=1, rowspan=4)

testImage = Label(root, text="Test Image", font=("Arial",30))
testImage.grid(row=1, column=1)

# Closest Result Frame
insertFrame3 = Frame(root,width=510,height=610, bg="#FFB9B9")
insertFrame3.grid(row=1,column=2, rowspan= 4)

closestResult = Label(root, text="Closest Result", font=("Arial",30))
closestResult.grid(row=1, column=2)

# button_quit = Button(root, text="Exit Program", command=root.quit)
# button_quit.pack()
root.mainloop() 