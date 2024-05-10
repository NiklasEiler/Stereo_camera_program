import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from scipy.stats import mode

count= 1
class Bildverarbeitung:
    def __init__(self, Img):
        # Bilder
        try:
            self.img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)

        except:
            self.img = Img

        self.img_shadow_re = self.img.copy()

        #finden der region of interest / bauteil
        x, y, crop_size = 640, 400, 250 #größe und position
        self.crop = self.img[y:y+crop_size, x:x+crop_size].copy()

        self.imgbin = np.zeros_like(self.crop)
        self.imgregion = np.zeros_like(self.crop)
        
        #zwischen Bilder
        self.emty = np.zeros_like(self.crop)
        self.countur_img = cv2.cvtColor(self.crop, cv2.COLOR_GRAY2BGR)
        self.aussenkontur_img = np.zeros_like(self.crop)
        self.edges = np.zeros_like(self.crop)
        self.imgclean = self.crop.copy()
        self.weisserstrich= self.crop.copy()

        

        # Daten
        self.flaechemittekoor = np.zeros(2)
        self.weisserstrichkoor = np.zeros(2) 
        self.mittekoor = np.zeros(2)
        self.flaeche = -1
        self.max_grad = -1
        self.dis_grad = np.zeros(50)
        self.dis_camera = -1
        self.metadata = "Emty"
        self.max_radius = 1000
        self.min_radius=0
        self.median_radius=0
        self.mittelwert_radius=0
        self.var_radius=0
        self.data=np.zeros(22) #Fläche, max_grad, min_grad, var Grad, mittel Grad, median Grad, rundheit, dis fl. mitte zur mitte, winkel weis, d1 , d2 ,d3 
        

        # Error
        self.error = "Erros: "
        
    def binar(self, vis=False): 
        global count
        rio= self.crop.copy()
        

        #Grauwert anpassung       
        rio[rio > 245] = 245
        
        #median filter gegeb rauschen
        rio=cv2.medianBlur(rio, 21)
        
        
        #Bin 
        ret, self.imgbin = cv2.threshold(cv2.bitwise_not(rio), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # Dimand Kernel
        size = 5
        kernel = np.zeros((size, size), dtype=np.uint8)
        # Define the center of the kernel
        center = size // 2        
        for i in range(size):
            for j in range(size):
                if abs(i - center) + abs(j - center) <= center:
                    kernel[i, j] = 1

        #dilation und erosion
        self.imgbin = cv2.morphologyEx(self.imgbin, cv2.MORPH_OPEN, kernel)
        self.imgbin = cv2.morphologyEx(self.imgbin, cv2.MORPH_CLOSE, kernel)

        #Konturfilter
        contours, _ = cv2.findContours(self.imgbin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iteriere durch die Konturen und filtere Regionen basierend auf der Fläche
        threshold_area_small = 600
        threshold_area_big = 60000
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter != 0:
                if area > threshold_area_small and area < threshold_area_big :
                    cv2.drawContours(self.imgregion, [
                                     contour], 0, 255, thickness=cv2.FILLED)
        
        
        #maske nutzen um Hintergrund wegzuschneiden
        self.imgclean[self.imgregion == 0] = 255

        if vis:
            cv2.imshow("rio", rio)
            cv2.waitKey(0)
            cv2.imshow("region", self.imgregion)
            cv2.waitKey(0)
            
            cv2.imshow("clean", self.imgclean)
            cv2.waitKey(0)

        #Berechnung von Eigenschaftejn
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
            
        roundness = (4 * np.pi * area) / (perimeter*perimeter)
        self.data[0] = cv2.countNonZero(self.imgregion)
        self.data[7] = roundness
        self.data[8] = perimeter
         
    def flaechen_mittelpunkt(self):
        # Berechne den gewichteten Durchschnitt der Koordinaten
        Mr = cv2.moments(self.imgregion)

        # Koordinaten berechnen und in Kounturbild einfügen
        try:
            self.flaechemittekoor[0] = int(Mr["m10"] / Mr["m00"])
            self.flaechemittekoor[1] = int(Mr["m01"] / Mr["m00"])
            self.data[11]=self.flaechemittekoor[0]
            self.data[12]=self.flaechemittekoor[1]
            size = 10
            thickness = 3
            cv2.line(self.countur_img, (int(self.flaechemittekoor[0] - size/2), int(self.flaechemittekoor[1])), (int(
                self.flaechemittekoor[0] + size/2), int(self.flaechemittekoor[1])), (140, 191, 232), thickness)
            cv2.line(self.countur_img, (int(self.flaechemittekoor[0]), int(self.flaechemittekoor[1] - size/2)), (int(
                self.flaechemittekoor[0]), int(self.flaechemittekoor[1] + size/2)), (140, 191, 232), thickness)
        except:
            self.error = self.error + ", Flächenmitte nicht gefunden"

    def aussenkontur(self, vis= False):
        #Kreuzkernel
        kernel_size = 3
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        center = kernel_size // 2
        kernel[:, center] = 1
        kernel[center, :] = 1
        
        #Dilatation und Erosion auf globales Binärbild
        img_erosion = cv2.erode(self.imgregion, kernel, iterations=1)
        img_dilation = cv2.dilate(self.imgregion, kernel, iterations=1)
        self.aussenkontur_img = cv2.subtract(img_dilation, img_erosion)
        #Kontur finden
        contours, _ = cv2.findContours(
            self.aussenkontur_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.countur_img, contours, -1, (45, 38, 161), 2)   	
        
        #Wenn Kontur gefunden maximalen Radius und länge bestimmen
        if len(contours) != 0:
            perimeter = cv2.arcLength(contours[0], True)
            self.max_radius = perimeter/(2*np.pi)

        if vis:
            cv2.imshow("aussenkontur", self.aussenkontur_img)
            cv2.waitKey(0)

    def mittelpunktzuaussen(self):
        #Kontur finden
        countur_aussen, _ =cv2.findContours( self.aussenkontur_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #Gratabstand Matrix defenieren
        dis_grad= np.zeros((np.shape(countur_aussen)[1],3))
        
        #Koordinaten zuordnen
        i=0
        for point in countur_aussen[0]:
            dis_grad[i,0], dis_grad[i,1] = point[0]
            i+=1

        #Distanzen berechnen
        delta_y= dis_grad[:, 1] - self.mittekoor[0]
        delta_x= dis_grad[:, 0] - self.mittekoor[1]
        delta_d= np.vstack((delta_x,delta_y))
        dis_grad[:,2] = np.linalg.norm(delta_d, ord=None, axis=0, keepdims=False)
        max_grad=dis_grad[dis_grad[:,2]==max(dis_grad[:,2])][0]
        
        #Eigenschaften berechnen
        self.data[1]= np.max(dis_grad[2])
        self.data[2]= np.min(dis_grad[2])
        
        self.data[3]= np.var(dis_grad[2])
        self.data[4]= np.mean(dis_grad[2])
        self.data[5]= np.median(dis_grad[2])
        self.data[6]= mode(dis_grad[2]).mode

        return dis_grad, max_grad

    def bauteil_mitte(self):
        nnnn=1
        # adatives Binar Bild um Kreisregion besser raus zu kriegen
        clean = cv2.bitwise_not(cv2.adaptiveThreshold(cv2.bilateralFilter(
            self.imgclean, 5, 175, 175), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2))
        self.clean = clean.copy()
        # Maximalen Radius festlegen durch die Aussenkontur
        mr = int(self.max_radius)


        contours, hierarchy = cv2.findContours(
            clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter die Konturen nach Flaeche raus
        contours = [
            contour for contour in contours if cv2.contourArea(contour) >= 300]
        jkl=1
        # Sucht nach Kreisen im Kountur Bild und benutzt dafür jedes konturbild einzeln
        x, y, n, j = 0, 0, 0, 0
        circle_accept = False
        mp = np.zeros((50, 3))
        for cnt in contours:
            temp = np.zeros_like(self.crop)
            hough_temp = np.zeros_like(self.crop)
            temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(temp, [cnt], 0, (0, 0, 255), thickness=cv2.FILLED)
            hough_temp = temp[:, :, 2]



            # findet Kreise in den einzel Kounturen
            circles = cv2.HoughCircles(
                hough_temp,
                cv2.HOUGH_GRADIENT,
                dp=1,           # Inverse ratio of accumulator resolution to image resolution
                minDist=100000,     # Minimum distance between detected circles
                param1=1,      # Canny edge detection threshold
                param2=1,      # Accumulator threshold for circle detection
                minRadius=20,   # Minimum radius of the circles
                maxRadius=mr   # Maximum radius of the circles
            )


            

            # Filter die Kreise raus die nicht in die Region der der findlargesregion passen
            if circles is not None:
                circles = np.uint16(np.around(circles))

                for i in circles[0, :]:
                    check = np.zeros_like(self.imgregion)
                    cv2.circle(check, (i[0], i[1]), i[2], 255, -1)
                    test = cv2.subtract(check, self.imgregion)
                    if not test.any() != 0 and j <= 50:
                        n += 1
                        mp[j, 0] = i[0]
                        mp[j, 1] = i[1]
                        mp[j, 2] = i[2]
                        y += i[1]
                        x += i[0]
                        circle_accept = True
                        j = j+1
            else:
                self.error = self.error + ", Keine Radien gefunden"

            del temp
            del hough_temp

        mp = mp[~np.all(mp == 0, axis=1)]
        no_outliers = np.zeros_like(mp)

        # Sortiert die Mittelpunkte aus die zuweit weg von den anderen Mittelpunkt liegen
        #IQR (Interquartile Range) Method
        multiplier = 0.05
        run = True
        counter = 0
        while run and len(mp) != 0:
            q1 = np.percentile(mp[:,0:1], 45, axis=0)
            q3 = np.percentile(mp[:,0:1], 65, axis=0)
            iqr = q3 - q1
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            no_outliers = mp[((mp >= lower_bound) & (
                mp <= upper_bound)).all(axis=1)]
            if no_outliers.shape[0] >= 2:
                run = False
            elif counter > 250:
                run = False
                self.error = self.error + ", Keine gemeinsames Mittelpunktcluster gefunden bei Culster"
                no_outliers=mp
            else:
                multiplier += 0.01
                counter += 1

        no_outliers = no_outliers[~np.all(no_outliers == 0, axis=1)]

        #Distanz Außreißer finden
        run=True
        tmp_no_outliers=no_outliers
        counter=0
        min_d=1
        enter=False
        while run and len(no_outliers) != 0 :
            tmp_y = int(np.mean(tmp_no_outliers[:, 1]))
            tmp_x = int(np.mean(tmp_no_outliers[:, 0]))
            delta_y= no_outliers[:, 1] - tmp_y
            delta_x= no_outliers[:, 0] - tmp_x
            delta_d= np.vstack((delta_x,delta_y))
            distances = np.linalg.norm(delta_d, ord=None, axis=0, keepdims=False)
            if len(no_outliers[distances< min_d]) != 0:
                tmp_no_outliers=no_outliers[distances< min_d]
                enter=True

            if tmp_no_outliers.shape[0] >= 2 and enter:
                run = False
                no_outliers=tmp_no_outliers
            elif counter > 1000 :
                run = False
                self.error = self.error + ", Keine gemeinsames Mittelpunktcluster gefunden bei Distanze"
            else:
                min_d += 0.01
                counter += 1
            
        # Ein zeichnen der gefundene Kreise und Mittelpunkte und dann den gemittelten Mittelpunkt bestimmen und einzeichnen
        if circle_accept and len(mp) != 0 and len(no_outliers) != 0:
            if False:#Auswählen ob gefilterte Kreise oder alle eingefügt werden in Konturbild
                for c in mp:
                    cv2.circle(self.countur_img, (int(c[0]), int(
                        c[1])), int(c[2]), (204, 50, 153), 2)
                    cv2.circle(self.countur_img,
                               (int(c[0]), int(c[1])), 2, (0, 255, 255), 3)
            else:
                for c in no_outliers:
                    
                    cv2.circle(self.countur_img, (int(c[0]), int(
                        c[1])), int(c[2]), (0, 255, 0), 2)
                    cv2.circle(self.countur_img,
                            (int(c[0]), int(c[1])), 2, (0, 0, 255), 3)
                    #print(c[2])

            
            for c in no_outliers:
                if c[2] > 22 and c[2] < 28:
                    self.data[19] = 1  
                if c[2] > 40 and c[2] < 46:
                    self.data[20] = 1  
                if c[2] > 48 and c[2] < 54:
                    self.data[21] = 1  
                
            #Bauteilmitte berechnen
            y = int(np.mean(no_outliers[:, 1]))
            x = int(np.mean(no_outliers[:, 0]))
            self.mittekoor = np.array((y, x))
            self.data[9]= x
            self.data[10]= y
            self.data[15]= np.linalg.norm(np.array((x,y)) -self.flaechemittekoor)
            
            cv2.circle(self.countur_img, (x, y), 2, (0, 255, 0), 3)
            self.dis_grad, self.max_grad = self.mittelpunktzuaussen()
            cv2.circle(self.countur_img, (int(self.max_grad[0]), int(
                self.max_grad[1])), 2, (255, 0, 0), 3)
            cv2.line(self.countur_img, (int(self.max_grad[0]), int(self.max_grad[1])), (int(
                self.mittekoor[1]), int(self.mittekoor[0])), (0, 165, 255), 2)
            

        else:
            self.error = self.error + ", nicht möglich Mittelpunkte, Kreise, Maximale Distanz und maxilameln Distanzpunkt einzuzeihnen"
        del mp, no_outliers

    def weisserstrich_finden(self):
        #Hintergrunf entfernen
        self.weisserstrich[self.imgregion == 0] = 0
        #Tolranzband des Weißbereiches
        max=np.max(self.weisserstrich)-15
        #wenn nicht weiß genug nicht Detektieren
        if max>230:
            #Binärbild der weißen Bereiche
            _, self.weisserstrich  = cv2.threshold(self.weisserstrich, max, 255, cv2.THRESH_BINARY)
            # Berechne den gewichteten Durchschnitt der Koordinaten
            Mr = cv2.moments(self.weisserstrich)

            

            if Mr["m01"]!=0 and  Mr["m10"] !=0 and Mr["m00"] !=0: 
                #Koordinaten Weißen Striches berechnen
                self.weisserstrichkoor[0] = int(Mr["m10"] / Mr["m00"])
                self.weisserstrichkoor[1] = int(Mr["m01"] / Mr["m00"])
                self.data[13]=self.weisserstrichkoor[0]
                self.data[14]=self.weisserstrichkoor[1]
                #Abstände berechnen
                self.data[16]= np.linalg.norm(np.array((self.mittekoor[1],self.mittekoor[0])) -self.weisserstrichkoor)
                self.data[17]= np.linalg.norm(self.weisserstrichkoor -self.flaechemittekoor)
                size = 10
                thickness = 3
                
                #In Bildverarbeitungsvisualisierung einzeichnen
                cv2.line(self.countur_img, (int(self.weisserstrichkoor[0] - size/2), int(self.weisserstrichkoor[1])), (int(
                    self.weisserstrichkoor[0] + size/2), int(self.weisserstrichkoor[1])), (255, 50, 255), thickness)
                cv2.line(self.countur_img, (int(self.weisserstrichkoor[0]), int(self.weisserstrichkoor[1] - size/2)), (int(
                    self.weisserstrichkoor[0]), int(self.weisserstrichkoor[1] + size/2)), (255, 50, 255), thickness)
                
                #Vektoren erstellen
                vec_g=np.zeros_like(self.weisserstrichkoor)
                temp= cv2.cvtColor(self.weisserstrich, cv2.COLOR_GRAY2BGR)
                vec_w=self.weisserstrichkoor-self.mittekoor
                
                vec_g[0]=self.max_grad[1]-self.mittekoor[0]
                vec_g[1]=self.max_grad[0]-self.mittekoor[1]

                #Winkel berechnen
                self.data[18]=self.angle_between_vectors(vec_w, vec_g)

            else:
                self.error = self.error + ", Weisserstrich nicht gefunden"
        else:
            self.error = self.error + ", Weisserstrich nicht gefunden"
        
    def reduce_shadows(self, alpha, beta):
        # KKontrast erhöhen
        adjusted_image = cv2.convertScaleAbs(self.img, alpha=alpha, beta=beta)

        # Wert auf 0 bis 255 ziehen
        self.img_shadow_re = np.clip(adjusted_image, 0, 255)
        self.img = self.img_shadow_re.copy()

    def angle_between_vectors(self,v1, v2):
        #berechnung des winkels mittels Punktprodukt Formel
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)
        cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
        angle_radians = np.arccos(cos_theta)
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees

    def plot_histogram(self, gray_img):
        # Calculate histogram
        histogram = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

        # Plot histogram
        plt.figure()
        plt.title("Grayscale Image Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.plot(histogram, color='black')
        plt.xlim([0, 256])
        plt.show()

def natural_sort_key(s):
    # Regulärer Ausdruck zur Aufteilung der Zeichenkette in numerische und nicht-numerische Teile
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def image_it( image_folder, filename, speicher_img=False, vis=False, speichern_data=False):
    #bild laden
    img = cv2.imread(image_folder +'/'+ filename)
    img = Bildverarbeitung(img)
    #Bildverarbeitung schritte durch gehen
    try:
        img.reduce_shadows(1.2, 5) 
    except:
        print('Error Schadow')
    try:
        img.binar()
    except:
        print('Error Bin')
    try:
        img.flaechen_mittelpunkt()
    except:
        print('Error FM')
    try:
        img.aussenkontur()
    except:
        print('Error Aussenkontur')
    try:
        img.bauteil_mitte()
    except:
        print('Error BM')
    

    if vis:
        cv2.imshow("Kontur", img.countur_img)
        cv2.waitKey(0)
    
        
    if speicher_img:
        #cv2.imwrite('crop/crop' + str(i+691) + '.png',img.crop)
        cv2.imwrite('Bildverarbeitungtest/normal/' + filename + '.png',img.countur_img)
    
def main():
    #Bildordner vorgeben
    image_folder="C:/LUH/Master/Masterarbeit/Stereo_camera_program/Daten/glueh/r"
    #Ausgabe von Bildordner zur Überprufung
    print(os.listdir(image_folder))
    #Iteration durch alle Dateien
    for filename in sorted(os.listdir(image_folder), key=natural_sort_key): 
        #Bild Datei erkennen
        if filename.endswith('.jpg') or filename.endswith('.png'):
            #Bild laden zur überprüfung and path zusammen setzen
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)

            #Wenn bild da Bildverarbeitung ausführen
            if image is not None:
                print('###################################' + filename + '########################################')
                image_it(image_folder, filename, vis=False, speicher_img=True, speichern_data=False)
   
            else:
                print('Bild ' + filename + " nicht gefunden")
    
if __name__ == '__main__':
    main()
