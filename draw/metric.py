import numpy as np
import cv2

def bbox_overlaps(boxes,query_boxes): # boxes is the predicted boxes and query_boxes is the ground truth boxes
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

def metric_for_image(bbox=None,gt_bbox=None,label=None, gt_label=None, prob=None,final_mask=None):
    #find the overlaps between each predicted box and gt_box
    overlaps = bbox_overlaps(np.ascontiguousarray(bbox[:, :4], dtype=np.float),np.ascontiguousarray(gt_bbox[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)  #multiple bboxes may have a single GT

    max_overlaps = overlaps[np.arange(bbox.shape[0]), gt_assignment] #select the predicted boxes that are closest to the gt_box

    good = 0
    total_boxes = 0
    for i,overlap  in enumerate(max_overlaps):
        box = bbox[i]
        width = int(box[2])-int(box[0])
        height = int(box[3])-int(box[1])
        if prob[i,label[i]] > 0.5 and width*height >1000 and label[i]!=0: #eliminate if classification is less than 0.5. if the box is too small or the label is background
            total_boxes = total_boxes+1 #this will be the denominator
            if label[i] == gt_label[i]:
                if overlap >0.5: #if overlap of the BOXES is bigger than 0.5
                    output_mask = (final_mask[i] > 0.6).astype(np.uint8)

                    gt_maski = gt_mask[:,int(box[1]):int(box[3]),int(box[0]):int(box[2]),:] #crop from gt_mask given the predicted box
                    gt_maskii = np.zeros([112,112,7],np.uint8)
                    for x in range(7):
                        mask = gt_maski[...,x]
                        mask = mask[0]
                        gt_maskii[...,x] = cv2.resize(mask.astype(np.uint8),(112,112))

                    if IOU_mask(output_mask,gt_maskii) > 0.5: #if overlap of the MASKS is bigger than 0.5
                        good = good +1
    precision_over_image = float(good)/(float(total_boxes)+np.finfo(np.float32).eps)
    return precision_over_image

metrics = []
for i in range(0,512):
    bbox = np.load('data/bbox'+str(i)+'.npy')
    gt_bbox = np.load('data/gt_boxes'+str(i)+'.npy')
    final_mask = np.load('data/final_mask'+str(i)+'.npy')
    gt_label = np.load('data/gt_label'+str(i)+'.npy')
    image = np.load('data/image'+str(i)+'.npy')
    label = np.load('data/label'+str(i)+'.npy')
    prob = np.load('data/prob'+str(i)+'.npy')
    gt_mask = np.load('data/gt_mask'+str(i)+'.npy')
    metrics.append(metric_for_image(bbox,gt_bbox,label,gt_label,prob,final_mask))

print reduce(lambda x, y: x + y, metrics) / len(metrics)
