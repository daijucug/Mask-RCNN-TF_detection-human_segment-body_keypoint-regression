import numpy as np
import cv2
mask_targets = np.load("data/mask_targets.npy")
rois = np.load("data/rois.npy")
if len(mask_targets)!=len(rois):
    raise ArithmeticError

image = cv2.imread("data/testseg15_1.jpg")
for i in range(len(mask_targets)):
    roi= rois[i]
    image = cv2.rectangle(image,(roi[0],roi[1]),(roi[2],roi[3]),(255,255,0),3)
    cv2.imshow("image",image)
    cv2.waitKey(3000)
    combined = np.zeros((112,112),np.uint8)
    for x in range(7):
        maska = mask_targets[i,...,x].astype(np.uint8)
        combined = np.logical_or(combined,maska)
        cv2.imshow("maska",maska*255)
        cv2.waitKey(500)
    combined = combined.astype(np.uint8)
    cv2.imshow("maska",combined*255)
    cv2.waitKey(500)
#pentru celelalte 8 masti de aici nu exista gt pentru ca iou este mult prea mic. adica ele sunt 9 dar sunt initializate de la inceput cu 0
#uitate ca se face mask_targets doar pentru keep_inds
