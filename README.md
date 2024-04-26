# Stereo_camera_program
This Python program uses data from a stereo camera to analyze the images and the point cloud to estimate the height and features of the forged part. For that, it uses a U-net for the global binarisation and a hough transformation to find the middle of the part. As classifiers, it uses a random forest and a support vector machine. Moreover, it uses DBSCAN and k-means to find the heights of the part.


# Downloading the git reposetorie
	git clone https://github.com/NiklasEiler/Stereo_camera_program.git
	or download the diretory

# Installation for windows 
All inserted rows are terminal commands and should copied in the terminal.

1. install Python 3.11

3. open a console/terminal in the Stereo_camera_programm

2. create a Virtual Environment
	python3.11 -m venv venv

3. Activate the Virtual Environment 
	venv\Scripts\Activate.ps1
info: after the comand in front of the line should appear a (venv)
exmaple : (venv) (base) PS C:\LUH\test>

4.Install Dependencies
	pip install -r requirements.txt

# Start programm
	
