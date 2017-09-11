import numpy as np
from utils import draw_bbox_better,draw_bbox,draw_bbox_better_v2,draw_segmentation_parts,visualize_mask_gt,draw_human_body_parts
import os

arrays = os.listdir('/home/alex/PycharmProjects/data_keypoints/')
for x in arrays:
    print (x)
    array = np.load("/home/alex/PycharmProjects/data_keypoints/"+x)
    image = array[0]
    bbox = array[1]
    label =array[2]
    prob = array[3]
    gt_bbox = array[4]
    gt_label  = array[5]
    final_mask = array[6]
    gt_mask = array[7]
    keyp = array[8]
    gt_keyp = array[9]


    #visualize_mask_gt(bbox,final_mask,gt_mask,label,prob)
    #draw_segmentation_parts(1,image,name="seg"+str(x),bbox=bbox,label=label,gt_label=gt_label,prob=prob,final_mask=final_mask)
    #draw_bbox_better(1,image,name="seg"+str(x),bbox=bbox,label=label,gt_label=gt_label,prob=prob,final_mask=final_mask) ############this is for voc independent body parts

    draw_human_body_parts(1,image,name="seg"+str(x),bbox=bbox,label=label,gt_label=gt_label,prob=prob,final_mask=final_mask,keyp=keyp,gt_keyp=gt_keyp)
