import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import scipy.misc

FLAGS = tf.app.flags.FLAGS
_DEBUG = False

def draw_img(step, image, name='', image_height=1, image_width=1, rois=None):
    #print("image")
    #print(image)
    #norm_image = np.uint8(image/np.max(np.abs(image))*255.0)
    norm_image = np.uint8(image/0.1*127.0 + 127.0)
    #print("norm_image")
    #print(norm_image)
    source_img = Image.fromarray(norm_image)
    return source_img.save(FLAGS.train_dir + 'test_' + name + '_' +  str(step) +'.jpg', 'JPEG')

colors = np.random.randint(5, size=(80, 3))


def draw_bbox_better(step, image, name='', image_height=1, image_width=1, bbox=None, label=None, gt_label=None, prob=None,final_mask=None):
    import cv2
    #source_img = Image.fromarray(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #b, g, r = source_img.split()
    #source_img = Image.merge("RGB", (r, g, b))
    #draw = ImageDraw.Draw(source_img)
    #color = '#0000ff'
    if bbox is not None:
        dictinary = {}

        for i, box in enumerate(bbox):
            if (prob[i,label[i]] > 0.5) and (label[i] > 0):
                area = float((box[2]-box[0])*(box[3]-box[1]))
                while area in dictinary:
                    area+=1
                width = int(box[2])-int(box[0])
                height = int(box[3])-int(box[1])
                mask = final_mask[i]
                mask = mask[...,label[i]]
                mask = scipy.misc.imresize(mask,(height,width))

                dictinary[round(area,4)]=(box,label[i],gt_label[i],prob[i,label[i]],mask,colors[label[i],:])
        sorted_keys = sorted(dictinary.iterkeys(),reverse=True)

        big_mask = np.zeros((image.shape[0],image.shape[1],len(bbox)),dtype=np.float32)

        i=0
        for key in sorted_keys:
            bo, _,_,_,msk,_= dictinary[key]
            big_mask[int(bo[1]):int(bo[3]),int(bo[0]):int(bo[2]),i] = msk
            i=i+1

        max_indices = np.argmax(big_mask,axis=2)
        for key in sorted_keys:
            bo, lab,gt_lab,_,_,col= dictinary[key]
            for x in range(int(bo[0]),int(bo[2])):
                for y in range(int(bo[1]),int(bo[3])):
                    _,_,_,_,_,col = dictinary.values()[max_indices[y,x]]
                    #print col
                    #print (image[y,x,0] )
                    image[y,x,...] = col
                    #hsv[y,x,0]=color[0]
                    #hsv[y,x,1]=hsv[y,x,1]*0.9
            text = cat_id_to_cls_name(lab)
            image = cv2.putText(image,text,(2+int(bo[0]),2+int(bo[1])), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2)
            if lab != gt_lab:
                c = (255,0,0)
            else:
                c = (0,0,255)
            image = cv2.rectangle(image,(int(bo[0]),int(bo[1])),(int(bo[2]),int(bo[3])),c,3)
    cv2.imwrite('output/est_imgs/test_' + name + '_' +  str(step) +'.jpg',image)







def draw_bbox(step, image, name='', image_height=1, image_width=1, bbox=None, label=None, gt_label=None, prob=None,final_mask=None):
    #print(prob[:,label])
    source_img = Image.fromarray(image)
    b, g, r = source_img.split()
    source_img = Image.merge("RGB", (r, g, b))
    draw = ImageDraw.Draw(source_img)
    color = '#0000ff'
    if bbox is not None:
        for i, box in enumerate(bbox):
            if label is not None:
                if prob is not None:
                    if (prob[i,label[i]] > 0.5) and (label[i] > 0):
                        if gt_label is not None:
                            text  = cat_id_to_cls_name(label[i]) + ' : ' + cat_id_to_cls_name(gt_label[i])
                            if label[i] != gt_label[i]:
                                color = '#ff0000'#draw.text((2+bbox[i,0], 2+bbox[i,1]), cat_id_to_cls_name(label[i]) + ' : ' + cat_id_to_cls_name(gt_label[i]), fill='#ff0000')
                            else:
                                color = '#0000ff'  
                        else: 
                            text = cat_id_to_cls_name(label[i])
                        #############################DRAW SEGMENTATION
                        width = box[2]-box[0]
                        height = box[3]-box[1]
                        #print (final_mask.shape)
                        mask = final_mask[i]
                        mask = mask[...,label[i]]
                        mask = scipy.misc.imresize(mask,(height,width))
                        mask_pil = Image.fromarray(mask)
                        source_img.paste(mask_pil,(int(box[0]),int(box[1])))
                        #draw.bitmap((int(box[0]),int(box[1])),mask_pil,fill='#00ffff')
                        draw.text((2+bbox[i,0], 2+bbox[i,1]), text, fill=color)
                        if _DEBUG is True:
                            print("plot",label[i], prob[i,label[i]])
                        draw.rectangle(box,fill=None,outline=color)
                        
                    else: 
                        if _DEBUG is True:
                            print("skip",label[i], prob[i,label[i]])
                else:
                    #############################DRAW GT SEGMENTATION
                    if final_mask is not None:
                        mask = final_mask[i]
                        mask_pil = Image.fromarray(mask)
                        mask_pil = mask_pil.crop([int(box[0]),int(box[1]),int(box[2]),int(box[3])])
                        source_img.paste(mask_pil,(int(box[0]),int(box[1])))
                    text = cat_id_to_cls_name(label[i])
                    draw.text((2+bbox[i,0], 2+bbox[i,1]), text, fill=color)
                    draw.rectangle(box,fill=None,outline=color)


    return source_img.save(FLAGS.train_dir + 'est_imgs/test_' + name + '_' +  str(step) +'.jpg', 'JPEG')

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
