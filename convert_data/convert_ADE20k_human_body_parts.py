import cv2
import numpy as np
import scipy.io as sio
import numpy.ma as ma
import logging
import traceback
import tensorflow as tf
from PIL import Image
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType


#the body_parts_dict is used for combining multiple labels into a single part (right upper leg and right lower leg become the same class. right leg)
body_parts_dict = {
    182:1,#head
    82:2,#bottom????
    4:2,#torso
    83:2,#back
    110:2,#neck
    148:3,#left upper arm
    73:3,#left shoulder
    150:3, #left hand
    219:3,#left lower arm
    69:4,#right upper arm
    152:4,#right shoulder
    71:4,#right hand
    151:5,#left lower leg
    149:5,#left foot
    72:6,#right lower leg
    70:6#right foot
}

# body_parts_dict = {
#     182:1,#head
#     82:2,#bottom????
#     4:2,#torso
#     83:2,#back
#     110:2,#neck
#     148:3,#left upper arm
#     73:3,#left shoulder
#     150:3, #left hand
#     219:3,#left lower arm
#     69:3,#right upper arm
#     152:3,#right shoulder
#     71:3,#right hand
#     151:5,#left lower leg
#     149:5,#left foot
#     72:5,#right lower leg
#     70:5#right foot
# }

#load data takes some form of annotations (provided from dataset) and return the correct form of annotation for tfrecords
def loadData(image,Oi,Om,Pm,size):
    if len(Pm.shape)>2:
        Pm = Pm[...,0]
    Pm = Pm.astype(np.uint8)#parts mask
    Oi = Oi.astype(np.uint8)#object instances #more explanation here: http://groups.csail.mit.edu/vision/datasets/ADE20K/
    Pm = cv2.resize(Pm,size)
    Oi = cv2.resize(Oi,size)
    Om = cv2.resize(Om,size)
    persons = (Om == 1831).astype(np.uint8) #find all person instances. person label is 1831
    H,W = image.shape[0],image.shape[1]


    Oi = ma.masked_array(Oi, mask=np.logical_not(persons)) # eliminate the non-person instances
    Oi = np.ma.filled(Oi, 0)
    gt_boxes = [] #will have shape: [N,x1,y1,x2,y2,cls]
    masks_instances = [] #shape: [N,H,W,7]

    for x in np.unique(Oi): #for each person
        if x ==0:
            continue
        per = (Oi == x).astype(np.uint8).copy() #get the mask for a person
        per_before_erosion = per.copy()
        per = cv2.erode(per,np.ones((3,3),np.uint8),iterations=2) # the image is badly annotated so I eliminate some artifacts
        _,contours,hierarchy = cv2.findContours(per, 1, 2) ######### from here
        if len(contours) ==0:
            continue
        x1=100000
        y1=100000
        x2=-10000
        y2=-10000
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            xw,yh = x+w,y+h
            if x <x1:
                x1 = x
            if y <y1:
                y1 = y
            if xw > x2:
                x2=xw
            if yh >y2:
                y2=yh
        if x2-x1<30 or y2-y1 <30:                           ######### to here i find the bbox of the person as the mask for the person might contain multiple blobs
            continue

        masks_for_person = np.zeros((H,W,7),dtype=np.uint8) # whole body + 6 parts
        masks_for_person[...,0] = per_before_erosion # the first mask is the body

        if True:#######  ONLY BODY

            Partsmask = ma.masked_array(Pm, mask=np.logical_not(per))
            Partsmask = np.ma.filled(Partsmask, 0)
            Parts = Partsmask >0
            torso = cv2.erode(per - Parts,np.ones((5,5),np.uint8),iterations=2)# the torso is not annotated. so i remove from the whole body, the parts=> torso
            _,contours,hierarchy = cv2.findContours(torso.copy(), 1, 2)
            torso2 = np.zeros(shape=torso.shape,dtype=np.uint8)
            maxArea,mx,my,mw,mh=0,0,0,0,0
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                if(w*h>maxArea):
                    maxArea = w*h
                    mx,my,mw,mh = x,y,w,h
            torso2[my:my+mh,mx:mx+mw] = torso[my:my+mh,mx:mx+mw] ## again the torso is badly annotated that results in multiple blobs. so I select the biggest blob
            for p in np.unique(Partsmask): # the parts mask contain labels that are not parts. so i skip over those labels
                if p not in body_parts_dict.keys():
                    continue

                part = (Partsmask == p).astype(np.uint8) #select one body part
                masks_for_person[...,body_parts_dict[p]] = np.logical_or(masks_for_person[...,body_parts_dict[p]],part) #this is where I combine for example right upper leg and right lower leg
            masks_for_person[...,2] = np.logical_or(masks_for_person[...,2],torso2) # here is the torso
            if float(np.sum(masks_for_person))/float(H*W*7) < 0.0001: # sometimes the object instance mask exists but there is no body parts annotation. so I skip over it
                continue
        gt_boxes.append([x1,y1,x2,y2,1])
        masks_instances.append(masks_for_person)
    if len(gt_boxes) ==0:
        return False,None,None,None,H,W
    masks_instances = np.array(masks_instances,dtype=np.uint8)
    gt_boxes = np.array(gt_boxes,dtype=np.float32)
    mask = masks_instances[0,:,:,1] # this mask is used for visualization in tensorboard
    return True,gt_boxes,masks_instances,mask,H,W

def _int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def _to_tfexample_coco_raw(image_id, image_data, label_data,
                           height, width,
                           num_instances, gt_boxes, masks):
  """ just write a raw input"""
  return tf.train.Example(features=tf.train.Features(feature={
    'image/img_id': _int64_feature(image_id),
    'image/encoded': _bytes_feature(image_data),
    'image/height': _int64_feature(height),
    'image/width': _int64_feature(width),
    'label/num_instances': _int64_feature(num_instances),  # N
    'label/gt_boxes': _bytes_feature(gt_boxes),  # of shape (N, 5), (x1, y1, x2, y2, classid)
    'label/gt_masks': _bytes_feature(masks),  # of shape (N, height, width)
    'label/encoded': _bytes_feature(label_data),  # deprecated, this is used for pixel-level segmentation
  }))

options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
record_filename = "data/out_human_and_body_parts_ade_20k_max640edge.tfrecord"
with tf.python_io.TFRecordWriter(record_filename, options=options) as tfrecord_writer:
    for i in range(22300):
        try:
            img = Image.open("data/ade20k/output_dir/file"+str(i)+".jpg")
            image = cv2.imread("data/ade20k/output_dir/file"+str(i)+".jpg")
            h,w = image.shape[0],image.shape[1]
            ratio = float(h)/float(w)
            if h >=w: # I resize it so that the maximum edge size is 640
                if h>640:
                    h=640
                    w=int(h/ratio)
            elif w>h:
                if w >640:
                    w=640
                    h=int(ratio*w)
            image = cv2.resize(image,(w,h))
            objects = sio.loadmat("data/ade20k/output_dir/objects"+str(i)+".mat")['objects']
            Oi = sio.loadmat("data/ade20k/output_dir/Oi"+str(i)+".mat")['Oi']
            Om = sio.loadmat("data/ade20k/output_dir/Om"+str(i)+".mat")['Om']
            parts = sio.loadmat("data/ade20k/output_dir/parts"+str(i)+".mat")['parts']
            Pi = sio.loadmat("data/ade20k/output_dir/Pi"+str(i)+".mat")['Pi']
            Pm = sio.loadmat("data/ade20k/output_dir/Pm"+str(i)+".mat")['Pm']
            img_id = i
            persons_exist,gt_boxes,masks_instances,mask,H,W = loadData(image,Oi,Om,Pm,(w,h))
            if not persons_exist:
                continue

            mask_raw = mask.tostring()

            img_raw = image.tostring()
            example = _to_tfexample_coco_raw(
                  img_id,
                  img_raw,
                  mask_raw,
                  H, W, gt_boxes.shape[0],
                  gt_boxes.tostring(), masks_instances.tostring())
            tfrecord_writer.write(example.SerializeToString())

        except BaseException as error:
            #logging.error(traceback.format_exc())
            print error
    tfrecord_writer.close()
