import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_bf
import os


class Bildverarbeitung:
    def __init__(self, Img):
        #Bilder
        try:
            self.img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
            
        except: 
            self.img = Img
        self.imgbin = np.zeros_like(self.img)
        self.imgregion = np.zeros_like(self.img)
        self.img_shadow_re = self.img.copy()
        
        
        self.emty=np.zeros_like(self.img)
        self.countur_img=cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        self.aussenkontur_img=np.zeros_like(self.img)
        self.edges= np.zeros_like(self.img)
        self.imgclean= self.img.copy()

        #Daten
        self.flaechemittekoor= np.zeros(2)
        self.mittekoor= np.zeros(2)
        self.flaeche=-1
        self.max_grad=-1
        self.dis_grad=np.zeros(50)
        self.dis_camera=-1
        self.metadata="Emty"

        #Error
        self.error= "Erros: "

    def binar(self):   #Otsu mit Gaußfilterung         
        ret ,self.imgbin= cv2.threshold(cv2.bitwise_not(self.img),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #self.imgbin = cv2.adaptiveThreshold(self.img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        #self.imgbin = cv2.adaptiveThreshold(self.img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    def find_large_regions(self, threshold_area_small, threshold_area_big):
        contours, _ = cv2.findContours(self.imgbin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iteriere durch die Konturen und filtere Regionen basierend auf der Fläche
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter= cv2.arcLength(contour,True)
            if perimeter != 0:
                roundness= (4 * np.pi * area)/ (perimeter*perimeter)
                if area > threshold_area_small and area < threshold_area_big and roundness > 0.7:
                    # Zeichne die Region auf das Ergebnisbild
                    cv2.drawContours(self.imgregion, [contour], 0, 255, thickness=cv2.FILLED)
        
        kernel_size = 3
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        center = kernel_size // 2  
        kernel[:, center] = 1
        kernel[center, :] = 1
        self.imgregion = cv2.erode(self.imgregion, kernel, iterations=5) 
        self.imgregion = cv2.dilate(self.imgregion, kernel, iterations=5)
        self.imgclean[self.imgregion == 0] = 255
        

    def flaechen_mittelpunkt(self):
        # Berechne den gewichteten Durchschnitt der Koordinaten
        Mr= cv2.moments(self.imgregion)
 
        # Koordinaten berechnen uznd in Kounturbild einfügen
        try:
            self.flaechemittekoor[0]= int(Mr["m10"] / Mr["m00"])
            self.flaechemittekoor[1] = int(Mr["m01"] / Mr["m00"])
            size=10
            thickness=3
            cv2.line(self.countur_img, (int(self.flaechemittekoor[0] - size/2), int(self.flaechemittekoor[1])), (int(self.flaechemittekoor[0] + size/2), int(self.flaechemittekoor[1])), (0, 255, 0), thickness)
            cv2.line(self.countur_img, (int(self.flaechemittekoor[0]), int(self.flaechemittekoor[1] - size/2)), (int(self.flaechemittekoor[0]), int(self.flaechemittekoor[1] + size/2)), (0, 255, 0), thickness)
        except:
            self.error= self.error + ", Flächenmitte nicht gefunden"

    def aussenkontur(self):
        kernel_size = 3
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        center = kernel_size // 2  
        kernel[:, center] = 1
        kernel[center, :] = 1

        img_erosion = cv2.erode(self.imgregion, kernel, iterations=1) 
        img_dilation = cv2.dilate(self.imgregion, kernel, iterations=1) 
        self.aussenkontur_img = cv2.subtract(img_dilation , img_erosion )      

    def distance_to_point(self):
        # Find the coordinates of non-zero elements in the binary image
        points = np.column_stack(np.where(self.aussenkontur_img))

        # Calculate the distance transform
        distance_transform = distance_transform_bf(self.aussenkontur_img)

        # Find the distance to the target point for each non-zero point
        distances = np.linalg.norm(points - self.mittekoor, axis=1)

        # Create a NumPy array with columns [x, y, distance]
        result_array = np.column_stack((points, distances))

        # Find the index of the row with the maximum distance
        max_distance_index = np.argmax(distances)

        # Extract coordinates of the point with the maximum distance
        max_distance_coord = result_array[max_distance_index, :]
        #pass Array an standart größe an Todo
        

        return result_array, max_distance_coord

    def bauteil_mitte(self, threshold_area_small, threshold_area_big ):
        self.clean = cv2.adaptiveThreshold(self.imgclean,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        self.clean = cv2.bitwise_not(self.clean)
        self.edges = cv2.Canny(self.clean, 50, 500 ,apertureSize = 5,  L2gradient = True )
        contours, _ = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iteriere durch die Konturen und filtere Regionen basierend auf der Fläche
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)    
            roundness = (4 * np.pi * area) / (perimeter * perimeter)
            if True:# roundness>0.6 : # area > threshold_area_small and area < threshold_area_big :
                # Zeichne die Region auf das Ergebnisbild
                cv2.drawContours(self.emty, [contour], 0, 255, thickness=1)

        
                
        circles = cv2.HoughCircles(
            self.edges,
            cv2.HOUGH_GRADIENT,
            dp=1,          # Inverse ratio of accumulator resolution to image resolution
            minDist=5e-324,    # Minimum distance between the centers of detected circles 5e-324
            param1=300,     # Upper threshold for the internal Canny edge detector
            param2=150,     # Threshold for center detection.
            minRadius=10,  # Minimum radius of the detected circles
            maxRadius=80  # Maximum radius of the detected circles
            
        )
        x, y, n= 0,0,0
        circle_accept=False
        # Mittepunbkt mitteln und gefundene Kreise einfügen ins Konturbild
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                if i[2] < 200 and i[2] > 0:
                    # Draw the outer circle
                    cv2.circle(self.countur_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    #cv2.circle(self.countur_img, (i[0], i[1]), 2, (0, 0, 255), 3)
                    n+=1
                    y+=i[1]
                    x+=i[0]
                    circle_accept= True
            if circle_accept:
                y= int(y/n)
                x=int(x/n)
                self.mittekoor= np.array((y,x))
                cv2.circle(self.countur_img, (x,y), 2, (0, 0, 255), 3)
                self.dis_grad, self.max_grad = self.distance_to_point()
                cv2.circle(self.countur_img, (int(self.max_grad[1]), int(self.max_grad[0])), 2, (255, 0, 0), 3)
                cv2.line(self.countur_img, (int(self.max_grad[1]), int(self.max_grad[0])), (int(self.mittekoor[1]), int(self.mittekoor[0])), (0, 165, 255), 2)
        else:
            self.error = self.error + "Bauteilmitte nicht gefunden"   
                


    def speichern(self, speicher_path, name, idx):
        cv2.imwrite('bilder/original'+ '_' + str(name) + str(idx) + '_' +'.png',self.img)
        cv2.imwrite('bilder/bin'+ '_' + str(name) + str(idx) + '_' +'.png',self.imgbin) 
        cv2.imwrite('bilder/region'+ '_' + str(name) + str(idx) + '_' +'.png',self.imgregion)
        cv2.imwrite('bilder/countur'+ '_' + str(name) + str(idx) + '_' +'.png',self.countur_img)
        cv2.imwrite('bilder/aussenkontur'+ '_' + str(name) + str(idx) + '_' +'.png',self.aussenkontur_img)
        cv2.imwrite('bilder/sobel'+ '_' + str(name) + str(idx) + '_' +'.png',self.edges)
        return 0
        #self.data
        #self.metadata
        ##self.mittekoor
        
    def reduce_shadows(self, alpha, beta):
            # Increase the image contrast
        adjusted_image = cv2.convertScaleAbs(self.img, alpha=alpha, beta=beta)

        # Clip pixel values to be in the valid range [0, 255]
        self.img_shadow_re= np.clip(adjusted_image, 0, 255)
        self.img=self.img_shadow_re.copy()

def main():
    
    for i in range(2):
        img = cv2.imread("bilder/original_right" + str(i+1) + "_.png")
        img= Bildverarbeitung(img)
        img.reduce_shadows(1.2,25)
        img.binar()
        img.find_large_regions(300, 100000)
        img.aussenkontur()
        img.bauteil_mitte(1,1000000000000000000)
        
        #img.flaechen_mittelpunkt()
        #img.Hough_Circles()
        #img.mitte()

        cv2.imshow("test_image", img.clean)
        cv2.waitKey(0)
        cv2.imshow("test_image", img.edges)
        cv2.waitKey(0)
        #cv2.imshow("test_image", img.emty)
        #cv2.waitKey(0)
        #cv2.imshow("test_image", img.img_shadow_re)
        #cv2.waitKey(0)
        cv2.imshow("test_image", img.emty)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imwrite('counturs' + str(i) + '.png',img.countur_img)
        del img



if __name__ == '__main__':
    
    main()
    
    
    

    