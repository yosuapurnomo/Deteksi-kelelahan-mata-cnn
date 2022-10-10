import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image, ImageTk
import cv2 as cv
from CnnProses import openClose as oC


class main:

    def __init__(self):
        self.labelClass = None
        self.textClass = None
        self.textFile = None
        self.label = None
        self.myImage = None
        self.img = None
        self.window = tk.Tk()
        self.path = None

    def openFile(self):
        self.path = filedialog.askopenfilename(initialdir="images", title="Select A File",
                                               filetypes=(("Jpg files", "*.jpg"),("png files", "*.png"),
                                                          ("all files", "*.*")))
        print(self.path)
        self.textFile.set(value=self.path)
        img = Image.open(self.path)
        img = img.resize((300, 300), Image.ANTIALIAS)
        tkImage = ImageTk.PhotoImage(img)
        self.myImage.config(image=tkImage)
        self.myImage.image = tkImage
        self.classification()

    def classification(self):
        run = oC()
        run.convSkenario2(self.path)
        run.setModelWeight()
        # print(np.argmax(run.data))
        if np.argmax(run.getDataPredict()[0]) == 0:
            Classi = "Mata Terbuka"
        else:
            Classi = "Mata Tertutup"
        print(Classi)
        self.textClass.set(value=Classi)

    def runWindow(self):
        self.window.title("Buka Tutup Mata")
        self.window.geometry("500x500")

        tk.Button(self.window, text="Load Image", command=self.openFile).pack(padx=100, pady=8)
        current_var = tk.StringVar()
        self.textFile = tk.StringVar()
        self.label = tk.Label(self.window, textvariable=self.textFile)
        self.label.pack()

        self.myImage = tk.Label(self.window)
        self.myImage.pack()
        self.textClass = tk.StringVar()
        self.labelClass = tk.Label(self.window, textvariable=self.textClass)
        self.labelClass.pack()

        self.window.mainloop()


main = main()
main.runWindow()