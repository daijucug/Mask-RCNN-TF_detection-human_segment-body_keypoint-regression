import numpy as np
from utils import draw_human_body_parts

for x in range(1,150):
    array = np.load("/home/alex/PycharmProjects/data/array"+str(x)+".npy")
    image = array[0]
    bbox = array[1]
    label =array[2]
    prob = array[3]
    gt_bbox = array[4]
    gt_label  = array[5]
    final_mask = array[6]
    gt_mask = array[7]

    #visualize_mask_gt(bbox,final_mask,gt_mask,label,prob)
    #draw_segmentation_parts(1,image,name="seg"+str(x),bbox=bbox,label=label,gt_label=gt_label,prob=prob,final_mask=final_mask)
    #draw_bbox_better(1,image,name="seg"+str(x),bbox=bbox,label=label,gt_label=gt_label,prob=prob,final_mask=final_mask) ############this is for voc independent body parts
    print (x)
    draw_human_body_parts(x,image,name="seg"+str(x),bbox=bbox,label=label,gt_label=gt_label,prob=prob,final_mask=final_mask)


