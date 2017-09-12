import numpy as np
import cv2
from utils import draw_human_body_parts

def bbox_overlaps(boxes,query_boxes):#predicted, ground_truth
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    iw, ih, box_area,ua,k, n = 0,0,0,0,0,0
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def IOU_mask(mask,gt_mask):
    intersection = np.sum( (mask * gt_mask) > 0 )
    union = np.sum((np.logical_or(mask,gt_mask))> 0)
    return float(intersection)/float(union+1)

import numpy.ma as ma
import scipy.misc
def metric_for_image(pindex=-1, bbox=None,gt_bbox=None,label=None, gt_label=None, prob=None,final_mask=None,gt_mask=None):
    #print (label, gt_label)

    overlaps = bbox_overlaps(np.ascontiguousarray(bbox[:, :4], dtype=np.float),np.ascontiguousarray(gt_bbox[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)  #multiple bboxes may have a single GT

    max_overlaps = overlaps[np.arange(bbox.shape[0]), gt_assignment] #if there are 3 proposal assigned for a single box, get the one that has the maximum overlap

    good = 0
    total_boxes = 0
    IOU_instances = []
    for i,overlap  in enumerate(max_overlaps):

        box = bbox[i]
        width = int(box[2])-int(box[0])
        height = int(box[3])-int(box[1])
        if prob[i,label[i]] > 0.5 and width*height >1000 and label[i]!=0:
            total_boxes = total_boxes+1
            if label[i] == gt_label[i]:
                if overlap > 0.5:
                    output_mask = (final_mask[i] > 0.6).astype(np.uint8)

                    gt_maski = gt_mask[:,int(box[1]):int(box[3]),int(box[0]):int(box[2]),:]
                    gt_maskii = np.zeros([112,112,7],np.uint8)
                    for x in range(7):
                        mask = gt_maski[...,x]
                        mask = mask[0]
                        gt_maskii[...,x] = cv2.resize(mask.astype(np.uint8),(112,112))

                    #body_mask = output_mask[...,0]
                    for x in range(1,7):
                        part = output_mask[...,x]
                        part = np.logical_and(part,output_mask[...,0])
                        part = ma.masked_array(output_mask[...,x], mask=np.logical_not(part))
                        part = np.ma.filled(part, 0)
                        output_mask[...,x] = part

                    output_mask[...,4] = np.zeros((112,112),np.uint8)
                    output_mask[...,6] = np.zeros((112,112),np.uint8)
                    if pindex==-1:iou = IOU_mask(output_mask,gt_maskii)
                    else: iou = IOU_mask(output_mask[...,pindex],gt_maskii[...,pindex])
                    IOU_instances.append(iou)
                    if iou > 0.5:
                        good = good +1
    print (IOU_instances)
    precision_over_image = float(good)/(float(total_boxes)+np.finfo(np.float32).eps)
    return precision_over_image

