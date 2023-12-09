from tkinter import *
from PIL import ImageTk, Image
import os
#import cv2


if __name__ == "__main__":
    #image1 = cv2.imread("C:\LUH\Master\Masterarbeit\Programm\test_b.png")


    root = Tk()
    img = ImageTk.PhotoImage(Image.open("test_b.png"))
    panel = Label(root, image = img)
    panel.pack(side = "bottom", fill = "both", expand = "yes")
    root.mainloop()