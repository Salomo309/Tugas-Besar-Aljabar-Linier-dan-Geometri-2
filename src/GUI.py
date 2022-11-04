from tkinter import *

root = Tk()

# Creating a Label Widget
# myLabel1 = Label(root, text="Halo Gais")
# myLabel2 = Label(root, text="Bapak Kau 10")

e = Entry(root)
e.pack()

def myClick() :
    myLabel = Label(root, text="Look! I clicked a button")
    myLabel.pack()

myButton = Button(root, text="Pencet aku wak", command=myClick, fg="blue")

# Showing it onto the screen
# myLabel1.grid(row=0, column= 10)
# myLabel2.grid(row=5, column=0)
myButton.pack()

root.mainloop()