import numpy as np
from utils import draw_bbox_better,draw_bbox,draw_bbox_better_v2,draw_segmentation_parts,visualize_mask_gt

for x in range(0,459,3):
    bbox = np.load('data/bbox'+str(x)+'.npy')
    final_mask = np.load('data/final_mask'+str(x)+'.npy')
    gt_label = np.load('data/gt_label'+str(x)+'.npy')
    image = np.load('data/image'+str(x)+'.npy')
    label = np.load('data/label'+str(x)+'.npy')
    prob = np.load('data/prob'+str(x)+'.npy')
    gt_mask = np.load('data/gt_mask'+str(x)+'.npy')

    #visualize_mask_gt(bbox,final_mask,gt_mask,label,prob)
    #draw_segmentation_parts(1,image,name="seg"+str(x),bbox=bbox,label=label,gt_label=gt_label,prob=prob,final_mask=final_mask)
    draw_bbox_better(1,image,name="seg"+str(x),bbox=bbox,label=label,gt_label=gt_label,prob=prob,final_mask=final_mask)


