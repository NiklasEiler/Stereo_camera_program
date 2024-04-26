import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def masked(n):
    for i in range(n):
        print(i+1)
        path =os.path.abspath(os.getcwd()) + "/umkreist/crop"+ str(i+1)+ ".png"
        img = cv2.imread(path)
        
        
        if img is not None:
            cut = img.copy()
            inv = img.copy()
            
            
            # Convert BGR to HSV (Hue, Saturation, Value)
            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Define range of yellow color in HSV
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])

            # Threshold the HSV image to get only yellow colors
            yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

            # Invert the mask to get non-yellow pixels
            non_yellow_mask = cv2.bitwise_not(yellow_mask)

            # Set non-yellow pixels to white
            img[non_yellow_mask != 0] = [255, 255, 255]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, global_bin = cv2.threshold(cv2.bitwise_not(gray), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            edges = cv2.Canny(global_bin, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blank = np.zeros_like(gray)
            cv2.drawContours(blank, contours, -1, 255, thickness=cv2.FILLED)

            cut[blank == 255] = [255, 255, 255]
            inv[blank != 255] = [255, 255, 255]

            cv2.imwrite("cut/"+ str(i+1)+ ".png",cut )
            cv2.imwrite("inverse/"+ str(i+1)+ ".png", inv)
            cv2.imwrite("maske/"+ str(i+1)+ ".png" , blank)
            original = cv2.imread("C:\LUH\Master\Masterarbeit\Stereo_camera_program\Daten\crop\crop" + str(i+1)+ ".png")
            
            cv2.imwrite("images/"+ str(i+1)+ ".png" , original)
            #cv2.imshow("org", img)
            #cv2.imshow("gray", gray)
            
            #cv2.imshow("bin", global_bin)
            #cv2.imshow("res", blank)
            #cv2.imshow("cut", cut)
            #cv2.imshow("inv", inv)
            #cv2.waitKey(0) 



def plot_histogram(image):
    # Split the image into its three color channels
    b, g, r = cv2.split(image)

    # Compute the histograms for each color channel
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

    # Plot histograms using Matplotlib
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.title("Blue Channel Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist_b, color='blue')

    plt.subplot(3, 1, 2)
    plt.title("Green Channel Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist_g, color='green')

    plt.subplot(3, 1, 3)
    plt.title("Red Channel Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist_r, color='red')

    plt.tight_layout()
    plt.show()


masked(200)

#plot_histogram(img)

