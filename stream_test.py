import cv2


#img= cv2.VideoCapture('rtsp://169.254.57.0')



#cv2.imshow('image window', img)
# add wait key. window waits until user presses a key
#cv2.waitKey(0)
# and finally destroy/close all open windows
#cv2.destroyAllWindows()

from time import sleep
from vimba import *

@ScopedLogEnable(LOG_CONFIG_INFO_CONSOLE_ONLY)

def print_device_id(dev , state ):
    msg = 'Device: {}, State: {}'.format(str(dev), str(state ))
    Log.get_instance (). info(msg)
vimba = Vimba.get_instance ()
vimba.register_camera_change_handler(print_device_id)
vimba.register_interface_change_handler(print_device_id)
with vimba:
    sleep (10)