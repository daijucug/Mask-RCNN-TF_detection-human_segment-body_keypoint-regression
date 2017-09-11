import numpy as np
import cv2
import scipy.misc

def bbox_overlaps(boxes,query_boxes):#predicted, ground_truth
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    iw, ih, box_area,ua,k, n = 0,0,0,0,0,0,0
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
    return float(intersection)/float(union)

def metric_for_image(bbox=None,gt_bbox=None,label=None, gt_label=None, prob=None,final_mask=None):
    overlaps = bbox_overlaps(np.ascontiguousarray(bbox[:, :4], dtype=np.float),np.ascontiguousarray(gt_bbox[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)  #multiple bboxes may have a single GT
    total_boxes = bbox.shape[0]
    max_overlaps = overlaps[np.arange(total_anchors), gt_assignment]

    good = 0
    for i,overlap  in enumerate(max_overlaps):
        if label[i] == gt_label[i]:
            if overlap >0.5:
                if IOU_mask(final_mask,gt_mask) > 0.5:
                    good = good +1
    precision_over_image = float(total_boxes)/float(good)
    return precision_over_image

for x in range(0,125,5):
    bbox = np.load('data/Alex/bbox'+str(x)+'.npy')
    gt_bbox = np.load('data/Alex/gt_boxes'+str(x)+'.npy')#astea nu exista momentan
    final_mask = np.load('data/Alex/final_mask'+str(x)+'.npy')
    gt_label = np.load('data/Alex/gt_label'+str(x)+'.npy')
    image = np.load('data/Alex/image'+str(x)+'.npy')
    label = np.load('data/Alex/label'+str(x)+'.npy')
    prob = np.load('data/Alex/prob'+str(x)+'.npy')
    gt_mask = np.load('data/Alex/gt_mask'+str(x)+'.npy')

