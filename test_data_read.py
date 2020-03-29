import numpy as np
from sklearn import preprocessing
import matplotlib as plt
import cv2
import os

#cwd = os.getcwd()
#files = os.listdir(cwd)
#print("Files in %r: %s" % (cwd, files))

#with open("/home/matthew/workspace/robotic_vision/RV_Data2/d1_0001.dat", 'r') as rv:
with open("/home/matthew/workspace/robotic_vision/RV_Data/Translation/Y3/frm_0030.dat", 'r') as rv:
    data = rv.read().split()
    floats = []
    for elem in data:
        try:
            floats.append(float(elem))
        except ValueError:
            pass
    #print(floats)
    floats = np.array(floats)
    floats = floats.reshape((-1,176))
    #depth = floats[0:144]
    depth = floats[432:576]    #amplitude
    depth = np.reshape(depth,(144,176))
    #cv2.imshow('depth_frame',depth)
    height, width = depth.shape[:2]
    normalized_data = preprocessing.normalize(depth)
    cv2.namedWindow('normalized',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('normlized',width,height)
    cv2.imshow('normalized',normalized_data)
    cv2.waitKey(0)
