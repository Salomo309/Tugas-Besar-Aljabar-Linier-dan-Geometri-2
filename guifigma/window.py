from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog


def btn_clicked():
    print("Button Clicked")


window = Tk()

window.geometry("1200x700")
window.configure(bg="#e6bdff")
global fileChosen
global folderChosen
fileChosen = False
folderChosen = False


def openFolder():
    global myFolder
    global folderChosen
    window.foldername = filedialog.askdirectory()
    foldername = window.foldername.split(
        '/')[len(window.foldername.split('/'))-1]
    canvas.itemconfig(
        cfo,
        text=foldername
    )


def openFile():
    global myImage
    global fileChosen
    window.filename = filedialog.askopenfilename(
        initialdir="ALGEO02-21063", title="Select an image", filetypes=(("JPG Files", "*.jpg"), ("PNG Files", "*.png"), ("All Files", "*.*")))
    filename = window.filename.split('/')[len(window.filename.split('/'))-1]
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


canvas = Canvas(
    window,
    bg="#e6bdff",
    height=700,
    width=1200,
    bd=0,
    highlightthickness=0,
    relief="ridge")
canvas.place(x=0, y=0)

cf = canvas.create_text(
    354.5, 401.5,
    text="No File Chosen",
    fill="#540097",
    font=("Poppins-Regular", int(16)))

img0 = PhotoImage(file=f"img0.png")
b0 = Button(
    image=img0,
    borderwidth=0,
    highlightthickness=0,
    command=openFolder,
    relief="flat")

b0.place(
    x=88, y=232,
    width=177,
    height=45)

img1 = PhotoImage(file=f"img1.png")
b1 = Button(
    image=img1,
    borderwidth=0,
    highlightthickness=0,
    command=openFile,
    relief="flat")

b1.place(
    x=88, y=379,
    width=177,
    height=45)

img2 = PhotoImage(file=f"img2.png")
b2 = Button(
    image=img2,
    borderwidth=0,
    highlightthickness=0,
    command=btn_clicked,
    relief="flat")

b2.place(
    x=548, y=544,
    width=177,
    height=45)

canvas.create_text(
    115, 529.5,
    text="None",
    fill="#540097",
    font=("Poppins-Regular", int(16)))

canvas.create_text(
    739.5, 627.5,
    text="00.00",
    fill="#540097",
    font=("Poppins-Regular", int(16)))

cfo = canvas.create_text(
    368.5, 254.5,
    text="No Folder Chosen",
    fill="#540097",
    font=("Poppins-Regular", int(16)))

# placeholder1_img = PhotoImage(file=f"placeholder.jpg")
# placeholder1 = canvas.create_image(
#     675, 360,
#     image=placeholder1_img)

# placeholder2_img = PhotoImage(file=f"placeholder.jpg")
# placeholder2 = canvas.create_image(
#     995, 360,
#     image=placeholder2_img)

background_img = PhotoImage(file=f"background.png")
background = canvas.create_image(
    599.5, 343.0,
    image=background_img)

# window.resizable(False, False)
window.mainloop()
