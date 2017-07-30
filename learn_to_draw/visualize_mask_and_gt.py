import scipy.io as sio
import numpy as np
import cv2

annotation = sio.loadmat('2008_003228.mat')
final_mask = np.load('data/final_mask'+'0'+'.npy')
bbox = np.load('data/bbox'+'0'+'.npy')


for i in range(4):
    obj = annotation['anno'][0]['objects'][0][0][i]
    if (obj['class']=='person'):
        parts = obj['parts'][0]
        contour_mask = np.zeros((375,500),dtype=np.uint8)
        for part,i in zip(parts,range(4)):
            mask = part['mask']
            cv2.imshow("mask",mask*255)
            cv2.waitKey(1000)
