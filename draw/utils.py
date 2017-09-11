import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import scipy.misc
import cv2
import numpy.ma as ma

FLAGS = tf.app.flags.FLAGS
_DEBUG = False


#not used
def draw_img(step, image, name='', image_height=1, image_width=1, rois=None):
    #print("image")
    #print(image)
    #norm_image = np.uint8(image/np.max(np.abs(image))*255.0)
    norm_image = np.uint8(image/0.1*127.0 + 127.0)
    #print("norm_image")
    #print(norm_image)
    source_img = Image.fromarray(norm_image)
    return source_img.save(FLAGS.train_dir + 'test_' + name + '_' +  str(step) +'.jpg', 'JPEG')


#label colors
colors = []
colors.append([180,255,255])
colors.append([150,255,255])
colors.append([120,255,255])
colors.append([90,255,255])
colors.append([60,255,255])
colors.append([30,255,255])
colors.append([0,255,255])



def draw_human_body_parts(step, image, name='', image_height=1, image_width=1, bbox=None, label=None, gt_label=None, prob=None,final_mask=None):
    import cv2
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_body = hsv.copy()
    if bbox is not None:
        dictinary = {} #key: area, value:[box,label,gt_label,prob,mask,color] #i create this dictionary in order to sort by area in order to draw smaller boxes in front
        for i, box in enumerate(bbox):
            width = int(box[2])-int(box[0])
            height = int(box[3])-int(box[1])
            #l=label[i]
            #p = prob[i,label[i]]
            if (prob[i,label[i]] > 0.5) and width*height >1000 and label[i]!=0: #eliminate some boxes. label is the predicted score
                area = float((box[2]-box[0])*(box[3]-box[1]))
                while area in dictinary: #i compute the area in order to draw smaller boxes in front
                    area+=1

                mask = final_mask[i]
                masks = np.zeros((height,width,7))
                body_mask = mask[...,0] > 0.6
                body_mask2 = np.array(body_mask,np.uint8)
                masks[...,0] = scipy.misc.imresize(body_mask2,(height,width))

                # cv2.imshow("body_mask",body_mask.astype(np.uint8)*255)
                # cv2.waitKey(3000)
                for x in range(1,7):
                    maska = mask[...,x] > 0.6 # if prop for a pixel is bigger than 0.6, draw it
                    # cv2.imshow("maska"+str(x),maska.astype(np.uint8)*255)
                    # cv2.waitKey(3000)
                    maska = np.logical_and(maska,body_mask) # clip the parts in order to fit inside the body. the body is better segmented
                    maska = ma.masked_array(mask[...,x], mask=np.logical_not(maska))
                    maska = np.ma.filled(maska, 0)
                    #maska = maska >0
                    maska = scipy.misc.imresize(maska,(height,width))

                    masks[...,x] = maska
                dictinary[round(area,4)]=(box,label[i],gt_label[i],prob[i,label[i]],masks,colors[label[i]])
        sorted_keys = sorted(dictinary.iterkeys(),reverse=True)
        # cv2.waitKey(6000)
        for key,i in zip(sorted_keys,range(len(sorted_keys))):
            bo, lab,gt_lab,_,mask,col= dictinary[key] #mask has shape [H,W,7]

            max_indices = np.argmax(mask,axis=2) # this is for when two parts masks are overlapping. there i select the part with the highest probability
            #max_indices is an array with size [H,W] and its values represent the per-pixel label of the parts
            for x in range(int(bo[0]),int(bo[2])):
                for y in range(int(bo[1]),int(bo[3])):

                    xm = x-(int(bo[0]))
                    ym = y-(int(bo[1]))
                    if mask[ym,xm,max_indices[ym,xm]] >0: #
                        hsv[y,x,0] = colors[max_indices[ym,xm]][0]
                        hsv[y,x,1] = 255

            for x in range(int(bo[0]),int(bo[2])):
                for y in range(int(bo[1]),int(bo[3])):

                    xm = x-(int(bo[0]))
                    ym = y-(int(bo[1]))
                    if(mask[ym,xm,0]==1):
                        hsv_body[y,x,0] = colors[0][0]
                        hsv_body[y,x,1] = 150

        hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        hsv_body = cv2.cvtColor(hsv_body, cv2.COLOR_HSV2RGB)
        i=0
        for key in sorted_keys:
            bo, lab,gt_lab,_,_,col= dictinary[key]
            c = (255,0,0)
            bo, lab,gt_lab,_,_,col= dictinary[key]
            text = cat_id_to_cls_name(lab)
            i=i+1
            hsv = cv2.rectangle(hsv,(int(bo[0]),int(bo[1])),(int(bo[2]),int(bo[3])),c,3)
            hsv = cv2.putText(hsv,text+' '+str(i),(2+int(bo[0]),2+int(bo[1])), cv2.FONT_HERSHEY_SIMPLEX,0.5, color =(255,255,255))
            hsv_body = cv2.rectangle(hsv_body,(int(bo[0]),int(bo[1])),(int(bo[2]),int(bo[3])),c,3)
            hsv_body = cv2.putText(hsv_body,text+' '+str(i),(2+int(bo[0]),2+int(bo[1])), cv2.FONT_HERSHEY_SIMPLEX,0.5, color =(255,255,255))
    #cv2.imwrite('test_' + name + '_' +  str(step) +'.jpg',image)
    cv2.imwrite('/home/alex/PycharmProjects/data/test_seg' + name + '_' +  str(step) +'.jpg',hsv)
    cv2.imwrite('/home/alex/PycharmProjects/data/test_hsv' + name + '_' +  str(step) +'.jpg',hsv_body)

def cat_id_to_cls_name(catId):
    cls_name = np.array(['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
