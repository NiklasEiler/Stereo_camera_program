import open3d as o3d
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os

def compute_mode(arr):
    unique, counts = np.unique(arr, return_counts=True)
    max_count_index = np.argmax(counts)
    return unique[max_count_index]

def numpy_array_to_pointcloud(numpy_array):
    # Create an Open3D PointCloud object
    pointcloud = o3d.geometry.PointCloud()

    # Convert the NumPy array to an Open3D PointCloud
    pointcloud.points = o3d.utility.Vector3dVector(numpy_array)

    return pointcloud

def find_high(n, speichern=False,histoz_vis= False, pc_vis=False):
    
    high= np.zeros((n,3))
    high[high==0]=-1
    
    for i in range(n):
        try:
            csv_file_path = os.path.abspath(os.getcwd()) + '/Daten/3d/' + str(i+1) + '.csv' 
            print(csv_file_path)
            data = pd.read_csv(csv_file_path)

            data=data.values
            
            #filter based on knowing the positionen in x and y
            lower_threshold_x=0.1
            mask_x = data[:, 0] <= lower_threshold_x 
            filtered_data = data[mask_x]

            threshold_y=0.05
            mask_y = filtered_data[:, 1] <= threshold_y
            filtered_data = filtered_data[mask_y]

            
            

            

            # KMeans Cluster on z values to finde ground 
            z_data = filtered_data[:,2]
            n_clusters = 2
            initial_centroids = np.array([[np.max(z_data)], [np.min(z_data)]])
            
            z_data = filtered_data[:,2].reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, n_init=1)

            kmeans.fit(z_data)

            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_

            ground_offset= np.max(centroids[:][0])
            high[i,0]=ground_offset
            ground_label = np.argmax(centroids[:][0])
            z_part = z_data[labels != ground_label]
            ground_p = z_data[labels == ground_label]
            mask_z = filtered_data[:,2] < np.min(ground_p)
            filtered_data = filtered_data[mask_z]

            
            
            tmp = filtered_data.copy()
            
            mask_mod1= tmp[:,2] <= np.min(tmp[:,2]) + 0.001
            f_z =  tmp[mask_mod1]
            h1= np.mean(f_z[:,2])
            high[i, 1] =ground_offset - h1 

            mask_del = np.all(np.isin(tmp, f_z, invert=True), axis=1)
            tmp = tmp[mask_del]
            mask_del = tmp[:,2] > h1 + 0.005
            tmp = tmp[mask_del]

            #find dorn high with dbscan and modus
            eps = 0.005  # maximum distance between two samples for one to be considered as in the neighborhood of the other
            min_samples = 40  # number of samples (or total weight) in a neighborhood for a point to be considered as a core point
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(tmp)
            
            u_labels = np.unique(labels[labels != -1])
            tmp_h= np.zeros(len(u_labels))

            if False:
                pointcloud = numpy_array_to_pointcloud(tmp)
                o3d.visualization.draw_geometries([pointcloud])
                pointcloud = numpy_array_to_pointcloud(tmp[labels==0])
                o3d.visualization.draw_geometries([pointcloud])
                pointcloud = numpy_array_to_pointcloud(tmp[labels==1])
                o3d.visualization.draw_geometries([pointcloud])
            
            #print(len(u_labels))
            if len(u_labels) == 1:
                high[i, 2] = ground_offset - np.mean(tmp[labels==0][:,2])
            
            else:
                var_x=np.zeros(len(u_labels))
                var_y=np.zeros(len(u_labels))
                var=np.zeros(len(u_labels))
                for j in range(len(u_labels)):
                    var_x[j]=np.var(tmp[labels==j][:,0])
                    var_y[j]=np.var(tmp[labels==j][:,1])
                    var[j] = np.linalg.norm([var_x, var_y])
                    
                    #pointcloud = numpy_array_to_pointcloud(tmp[labels==j])
                    #o3d.visualization.draw_geometries([pointcloud])

                
                choosen_cluster= np.argmax(var) 
                            
                test_1 = [tuple(row) for row in filtered_data]
                test_2 = [tuple(row) for row in tmp[labels==choosen_cluster]]

                # Find indices of rows in array1 that are not in array2
                indices_to_keep = [i for i, row in enumerate(test_1) if row not in test_2]
                            
                
                #pointcloud = numpy_array_to_pointcloud(filtered_data)
                #o3d.visualization.draw_geometries([pointcloud])
                #pointcloud = numpy_array_to_pointcloud(filtered_data[indices_to_keep])
                #o3d.visualization.draw_geometries([pointcloud])
            
                
                    

                
                h3 = tmp[labels==choosen_cluster][:,2]
                max_h3= np.max(h3)
                h3 = h3[h3 > max_h3- 0.0025]
                high[i, 2] = ground_offset - np.mean(h3)
                #print(max_h3, np.mean(h3))
                #print(high[i,:])
        except:
            print('Punktwolke '+ str(i+1) + ' nicht gefunden')
        


        #visual
        if histoz_vis:
            plt.hist(z_part, bins=2000, edgecolor='black')
            #plt.hist(ground_p, bins=2000, edgecolor='orange')
            #plt.hist(f_z, bins=2000, edgecolor='black')
            plt.title('Verteilung der Z-Achsen Höhenwerte')
            plt.xlabel('Z Höhe')
            plt.ylabel('Häufigkeit')
            plt.grid(True)
            plt.show()

        


        if pc_vis :
            #pointcloud = numpy_array_to_pointcloud(data)
            pointcloud = numpy_array_to_pointcloud(filtered_data)
            #pointcloud = numpy_array_to_pointcloud(f_z)
            #pointcloud = numpy_array_to_pointcloud(tmp)
            #pointcloud = numpy_array_to_pointcloud(highs)
            #pointcloud.estimate_normals()
            #mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pointcloud, depth=9, width=0, scale=1.1, linear_fit=False)

            # Visualize the mesh
            #o3d.visualization.draw_geometries([mesh])
            # Visualize the point cloud
            o3d.visualization.draw_geometries([pointcloud])
    

    if speichern:
            labels=['Messblatte', 'Schmiedeteilhöhe', 'Dornhöhe']
            index= index=np.arange(1,n+1,1)
            df = pd.DataFrame(high, columns=labels, index=index)
            df.to_excel('höhe_data.xlsx')
            del high
            print('save')
            

if __name__ == "__main__":
    find_high(690, speichern=True ,histoz_vis= False, pc_vis=False )


    

