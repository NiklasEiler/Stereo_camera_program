import cv2
import numpy as np
import win32gui
import win32con

# Global variables
drawing = False
points = []
start = True
img = None  # Initialize img as global variable
region = []
x = 100  # X position of the window
y = 100  # Y position of the window
width = 250  # Width of the window
height = 250  # Height of the window
# Mouse callback function
def draw_poly(event, x, y, flags, param):
    global points, drawing, img

    if event == cv2.EVENT_LBUTTONDOWN and start:
        drawing = True
        points.append((x, y))

    elif event == cv2.EVENT_LBUTTONDOWN and not start:
        if drawing:
            points.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = False
        cv2.fillPoly(img, [np.array(points)], 0)
        
        #cv2.imshow('Selected Region', img)
        points = []

def process_image():
    global img
    
    img[img > 0] = 255
    
    

def save_image(i):
    global img
    rezized_img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(i+1) + ".png", rezized_img)


for i in range(50):
    # Read the image
    img = cv2.imread("C:/LUH/Master/Masterarbeit/Stereo_camera_program/Daten/crop/crop" + str(i+1) + ".png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[img == 0] = 1
    cv2.namedWindow('Select Region')
    cv2.setMouseCallback('Select Region', draw_poly)
    scale_factor = 5
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    while True:
        cv2.imshow('Select Region', img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press ESC to exit
            break
        elif key == ord('1'):  # Press '1' to save image and process
            process_image()
            save_image(i)
            break

    cv2.destroyAllWindows()

'''''

'''''