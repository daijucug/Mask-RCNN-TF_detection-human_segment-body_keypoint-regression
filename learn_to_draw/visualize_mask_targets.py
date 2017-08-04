import numpy as np
import cv2
mask_targets = np.load("data/mask_targets.npy")

for i in range(len(mask_targets)):
    print (len(mask_targets))
    combined = np.zeros((112,112),np.uint8)
    for x in range(7):
        maska = mask_targets[i,...,x].astype(np.uint8)
        combined = np.logical_or(combined,maska)
        cv2.imshow("maska",maska*255)
        cv2.waitKey(500)
    combined = combined.astype(np.uint8)
    cv2.imshow("maska",combined*255)
    cv2.waitKey(500)
