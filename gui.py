from stereo_camera_programm_gui.ui_form import Ui_Widget
import sys
import All_in_one 
import cv2
import open3d as o3d
import csv
import numpy as np
import os
import re
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from PySide6.QtWidgets import QApplication, QWidget, QFileDialog
from PySide6.QtGui import QPixmap, QImage

def delete_rows_in_other_array(arr1, arr2):

    # Convert arrays to sets of tuples for efficient comparison
    set1 = {tuple(row) for row in arr1}
    set2 = {tuple(row) for row in arr2}
    
    # Find rows in arr1 that are not in arr2
    result_set = set1 - set2
    
    # Convert back to numpy array
    result = np.array(list(result_set))
    
    return result

def numpy_array_to_pointcloud(numpy_array):
    # Create an Open3D PointCloud object
    pointcloud = o3d.geometry.PointCloud()

    # Convert the NumPy array to an Open3D PointCloud
    pointcloud.points = o3d.utility.Vector3dVector(numpy_array)

    return pointcloud

def get_folder_names(path):
    #get path open it and put all file names in a array that gets returned
    folder_names = []
    for entry in os.listdir(path):
        if os.path.isdir(os.path.join(path, entry)):
            folder_names.append(entry)
    return folder_names

def natural_sort_key(s):
    #sort for stribgs with different size 
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_img_and_crop(n, x, y):
    count=0
    img_path= os.path.abspath(os.getcwd()) + '/Messung'
    folder_names = get_folder_names(img_path)
    for folder_name in folder_names:
        if folder_name != '3d' and folder_name != 'Ergebnisse':
                
            image_folder=img_path + "/" + str(folder_name)

            for filename in sorted(os.listdir(image_folder), key=natural_sort_key): 
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(image_folder, filename)
                    if count>=n:
                        img = cv2.imread(image_path)
                        crop_size = 250 
                        img = img[y:y+crop_size, x:x+crop_size].copy()
                        return img
                    count=count+1

def cv2_to_qpixmap(cv_img):
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width
    q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_img) 

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)
        
        self.ui.execut_pc_pro.clicked.connect(self.open_csv)

        self.ui.start_bt.clicked.connect(self.verarbeitung_messung)

        self.x_crop= 650
        self.y_crop= 400
        self.ui.x_input_value.textChanged.connect(self.update_x)
        self.ui.y_input_value.textChanged.connect(self.update_y)

        image=get_img_and_crop(1, self.x_crop, self.y_crop )
        pixmap_crop_img=cv2_to_qpixmap(image)
        self.ui.crop_img.setPixmap(pixmap_crop_img)
        self.ui.iph_logo_1.setPixmap(QPixmap(u"iph_logo.png"))
        self.ui.iph_logo_2.setPixmap(QPixmap(u"iph_logo.png"))

        self.selected_csv_file=''
        self.pc_path= os.path.abspath(os.getcwd()) + '/Messung/3d'
        self.pc_filenames= self.get_file_names(self.pc_path)
        self.ui.pc_file_list.addItems(self.pc_filenames)
        self.ui.choosen_pc_file.setText('Ausgewählt: ' + self.selected_csv_file)
        self.ui.pc_file_list.itemClicked.connect(self.get_csv_name)

        self.ui.refresch.clicked.connect(self.refresh_new_filenames)
        
    def update_x(self):
        self.ui.x_crop_value.setText('x: ' + self.ui.x_input_value.text())
        self.x_crop= int( self.ui.x_input_value.text())
        image=get_img_and_crop(1, self.x_crop, self.y_crop )
        pixmap_crop_img=cv2_to_qpixmap(image)
        self.ui.crop_img.setPixmap(pixmap_crop_img)

    def update_y(self):
        self.ui.y_crop_value.setText('x: ' + self.ui.y_input_value.text())
        self.y_crop= int(self.ui.y_input_value.text())
        image=get_img_and_crop(1, self.x_crop, self.y_crop )
        pixmap_crop_img=cv2_to_qpixmap(image)
        self.ui.crop_img.setPixmap(pixmap_crop_img)   

    def get_file_names(self, directory):
        # Check if the directory exists
        if not os.path.isdir(directory):
            print("Error: Directory '{}' does not exist.".format(directory))
            return []

        # Get all file names in the directory
        file_names = sorted(os.listdir(directory), key=natural_sort_key)
        

        return [file_name for file_name in file_names if file_name.endswith('.csv') and os.path.isfile(os.path.join(directory, file_name))]

    def get_csv_name(self, item):
        self.selected_csv_file=item.text()
        self.ui.choosen_pc_file.setText('Ausgewählt: ' + self.selected_csv_file)
        

    def open_csv(self):
        if self.selected_csv_file != '':
            pc = pd.read_csv(self.pc_path + '/' + self.selected_csv_file)
            pc=pc.values
            pc, steps_tacken =self.pc_verarbeitung(pc)

            pointcloud = numpy_array_to_pointcloud(pc)
            o3d.visualization.draw_geometries([pointcloud], window_name=self.selected_csv_file + steps_tacken )

        else:
            print('Keine Datei Ausgewählt')

    def pc_verarbeitung(self, points):
        steps_tacken=''
        place_sep=''
        tmp_points=points.copy()

        #rog
        lower_threshold_x=0.1
        mask_rog1 = tmp_points[:, 0] <= lower_threshold_x 
        tmp_points = tmp_points[mask_rog1]

        threshold_y=0.05
        mask_rog2 = tmp_points[:, 1] <= threshold_y
        tmp_points = tmp_points[mask_rog2]
        rog_pc=tmp_points.copy()

        #ground
        z_data = tmp_points[:,2]
        n_clusters = 2
        initial_centroids = np.array([[np.max(z_data)], [np.min(z_data)]])        
        z_data = tmp_points[:,2].reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, n_init=1)
        kmeans.fit(z_data)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        ground_label = np.argmax(centroids[:][0])
        z_part = z_data[labels != ground_label]
        ground_p = z_data[labels == ground_label]
        mask_ground = tmp_points[:,2] < np.min(ground_p)
        ground_pc= tmp_points[~mask_ground]
        tmp_points = tmp_points[mask_ground]
        
        #upper high
        tmp = tmp_points.copy()  
        mask_mod1= tmp[:,2] <= np.min(tmp[:,2]) + 0.001
        f_z =  tmp[mask_mod1]
        upper_pc=f_z.copy()
        h1= np.mean(f_z[:,2])
        mask_del = np.all(np.isin(tmp, f_z, invert=True), axis=1)
        tmp = tmp[mask_del]
        mask_del = tmp[:,2] > h1 + 0.005
        tmp = tmp[mask_del]
        

        #find dorn high with dbscan and modus
        eps = 0.005  
        min_samples = 40  
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(tmp)
                
        u_labels = np.unique(labels[labels != -1])

        if len(u_labels) != 1:
            var_x=np.zeros(len(u_labels))
            var_y=np.zeros(len(u_labels))
            var=np.zeros(len(u_labels))
            for j in range(len(u_labels)):
                var_x[j]=np.var(tmp[labels==j][:,0])
                var_y[j]=np.var(tmp[labels==j][:,1])
                var[j] = np.linalg.norm([var_x, var_y])
                            
            choosen_cluster= np.argmax(var) 
                                        
            h3 = tmp[labels==choosen_cluster][:,2]
            dorn_pc= tmp[labels==choosen_cluster]
            max_h3= np.max(h3)
            mask_dorn= h3 > max_h3- 0.0025
            h3 = h3[mask_dorn]

        not_ticked=True
        if self.ui.rog.isChecked():
            res=rog_pc
            steps_tacken = steps_tacken + " Region of Interrest"
            place_sep=','
            not_ticked=False
        else:
            res=points.copy()

        if self.ui.del_bottom.isChecked():
            res = delete_rows_in_other_array(res, ground_pc)
            steps_tacken = steps_tacken + place_sep + " Untergrund entfernt"
            place_sep=','
            not_ticked=False

        if self.ui.del_upper.isChecked():
            res = delete_rows_in_other_array(res, upper_pc)
            steps_tacken = steps_tacken + place_sep + " Obereschicht Entfernen"
            place_sep=','
            not_ticked=False

        if self.ui.del_dorn.isChecked():
            res = delete_rows_in_other_array(res, dorn_pc)
            steps_tacken = steps_tacken + place_sep + " Dorn entfern"
            place_sep=','
            not_ticked=False

        if self.ui.rest.isChecked():
            res=np.append(dorn_pc,upper_pc, axis=0)
            steps_tacken = steps_tacken + place_sep + " Rest Entfernt"
            not_ticked=False

        if  not_ticked: 
            res=points.copy()

        return res, steps_tacken

    def refresh_new_filenames(self):
        while self.ui.pc_file_list.count() > 0:
            item = self.ui.pc_file_list.takeItem(0)
            del item
        self.pc_filenames= self.get_file_names(self.pc_path)
        self.ui.pc_file_list.addItems(self.pc_filenames)

    def verarbeitung_messung(self):
        All_in_one.main(x=self.x_crop, y=self.y_crop)
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())