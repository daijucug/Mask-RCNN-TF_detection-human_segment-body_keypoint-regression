import numpy as np
from utils import draw_human_body_parts
import cv2

for x in range(1,200):
    array = np.load("/home/alex/PycharmProjects/data/array"+str(x)+".npy")
    image = array[0]
    bbox = array[1]
    label =array[2]
    prob = array[3]
    gt_bbox = array[4]
    gt_label  = array[5]
    final_mask = array[6]
    gt_mask = array[7]
    # keyp = array[8]
    # gt_keyp = array[9]
    print x
    draw_human_body_parts(1,image,name="seg"+str(x),bbox=bbox,label=label,gt_label=gt_label,prob=prob,final_mask=final_mask)


