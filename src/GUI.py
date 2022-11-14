from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog

root = Tk()
root.title("GUI")
# root.geometry("1530x780")
root["bg"] = "#FFDDD2"

# width = root.winfo_screenwidth()
# height = root.winfo_screenheight()
# # setting tkinter root size
# root.geometry("%dx%d" % (width, height))

# Frame(root, width=427, height=250, bg='#272727').place(x=0, y=0)
title = Label(root, text="Face Recognition Salomo Ganteng",
              justify='center', font='Helvetica 30 bold', border=40, bg="#FFDDD2")
title.grid(row=0, column=0, columnspan=3)
# line = Frame(root, width=1410, height=1,bg="black", border=100).pack()

# Insert Frame
insertFrame1 = Frame(root, width=210, height=610, bg="#FFB9B9")
insertFrame1.grid(row=1, column=0, rowspan=4)


def openFolder():
    global myFolder
    root.foldername = filedialog.askdirectory()
    foldername = root.foldername.split('/')[len(root.foldername.split('/'))-1]
    folder = Label(root, text=foldername, font=(16)).grid(row=2, column=0)


folderButton = Button(root, bg="white", text="Please Insert a Folder",
                      command=openFolder, justify="left", font=(30))
folderButton.grid(row=1, column=0)


def openFile():
    global myImage
    root.filename = filedialog.askopenfilename(
        initialdir="ALGEO02-21063", title="Select an image", filetypes=(("JPG files", "*.jpg"), ("All files", "*.*")))
    filename = root.filename.split('/')[len(root.filename.split('/'))-1]
    file = Label(root, text=filename, font=(
        16), bg="#FFB9B9").grid(row=4, column=0)

    img = Image.open(root.filename)
    myImage = ImageTk.PhotoImage(img)
    new_width = 256
    new_height = int(new_width * myImage.height() / myImage.width())
    resized = img.resize((new_width, new_height), Image.ANTIALIAS)
    myImage = ImageTk.PhotoImage(resized)

    insertImage = Label(image=myImage).grid(row=1, column=1, rowspan=4)


imageButton = Button(root, bg="white", text="Please Insert an Image",
                     command=openFile, justify="left", font=(30))
imageButton.grid(row=3, column=0)

# Test Image Frame
insertFrame2 = Frame(root, width=510, height=610, bg="#FFB9B9")
insertFrame2.grid(row=1, column=1, rowspan=4)

testImage = Label(root, text="Test Image",
                  font=("Helvetica", 20), bg="#FFB9B9")
testImage.grid(row=1, column=1)

# Closest Result Frame
insertFrame3 = Frame(root, width=510, height=610, bg="#FFB9B9")
insertFrame3.grid(row=1, column=2, rowspan=4)

closestResult = Label(root, text="Closest Result",
                      font=("Helvetica", 20), bg="#FFB9B9")
closestResult.grid(row=1, column=2)

# button_quit = Button(root, text="Exit Program", command=root.quit)
# button_quit.pack()
root.mainloop()
