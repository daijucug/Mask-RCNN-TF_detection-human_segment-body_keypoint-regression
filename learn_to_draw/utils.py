import numpy as np
import scipy.misc
import numpy.ma as ma

colors = []
colors.append([180,255,255])
colors.append([150,255,255])
colors.append([120,255,255])
colors.append([90,255,255])
colors.append([60,255,255])
colors.append([30,255,255])
colors.append([0,255,255])


def draw_human_body_parts(step, image, name='', bbox=None, label=None, gt_label=None, prob=None,final_mask=None):
    import cv2
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if bbox is not None:
        dictinary = {}
        for i, box in enumerate(bbox):
            width = int(box[2])-int(box[0])
            height = int(box[3])-int(box[1])
            if (prob[i,label[i]] > 0.48) and width*height >1000 and label[i]!=0:
                area = float((box[2]-box[0])*(box[3]-box[1]))
                while area in dictinary:
                    area+=1
                mask = final_mask[i]
                masks = np.zeros((height,width,7))
                body_mask = mask[...,0] > 0.6
                masks[...,0] = scipy.misc.imresize(np.array(body_mask,dtype=np.uint8),(height,width))
                for x in range(1,7):
                    maska = mask[...,x] > 0.6
                    maska = np.logical_and(maska,body_mask)
                    maska = ma.masked_array(mask[...,x], mask=np.logical_not(maska))
                    maska = np.ma.filled(maska, 0)
                    maska = scipy.misc.imresize(maska,(height,width))
                    masks[...,x] = maska
                dictinary[round(area,4)]=(box,label[i],gt_label[i],prob[i,label[i]],masks,colors[label[i]])
        sorted_keys = sorted(dictinary.iterkeys(),reverse=True)

        for key,i in zip(sorted_keys,range(len(sorted_keys))):
            bo, lab,gt_lab,_,mask,col= dictinary[key]
            max_indices = np.argmax(mask,axis=2)

            # for x in range(int(bo[0]),int(bo[2])):
            #     for y in range(int(bo[1]),int(bo[3])):
            #
            #         xm = x-(int(bo[0]))
            #         ym = y-(int(bo[1]))
            #         if(max_indices[ym,xm]==0):
            #             continue
            #         if mask[ym,xm,max_indices[ym,xm]] >0:
            #             hsv[y,x,0] = colors[max_indices[ym,xm]][0]
            #             hsv[y,x,1] = 255

            for x in range(int(bo[0]),int(bo[2])):
                for y in range(int(bo[1]),int(bo[3])):

                    xm = x-(int(bo[0]))
                    ym = y-(int(bo[1]))
                    if mask[ym,xm,0] >0:
                        hsv[y,x,0] = colors[0][0]
                        hsv[y,x,1] = 255

        hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        i=0
        for key in sorted_keys:
            bo, lab,gt_lab,_,_,col= dictinary[key]
            if lab != gt_lab:
                c = (0,0,255)
            else:
                c = (255,0,0)
            bo, lab,gt_lab,_,_,col= dictinary[key]
            text = cat_id_to_cls_name(lab)
            i=i+1
            hsv = cv2.rectangle(hsv,(int(bo[0]),int(bo[1])),(int(bo[2]),int(bo[3])),c,3)
            hsv = cv2.putText(hsv,text+' '+str(i),(2+int(bo[0]),2+int(bo[1])), cv2.FONT_HERSHEY_SIMPLEX,0.5, color =(255,255,255))
    cv2.imwrite('/home/alex/PycharmProjects/data/test' + name + '_' +  str(step) +'.jpg',hsv)

def cat_id_to_cls_name(catId):
    cls_name = np.array([  'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                       'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                       'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                       'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                       'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                       'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                       'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                       'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                       'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                       'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                       'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                       'scissors', 'teddy bear', 'hair drier', 'toothbrush'])
    return cls_name[catId]
