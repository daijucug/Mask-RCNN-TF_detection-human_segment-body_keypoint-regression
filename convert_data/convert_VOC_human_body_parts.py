import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
import numpy as np
from PIL import Image
import scipy.io as sio
import cv2
import traceback
import logging

#the body_parts_dict is used for combining multiple labels into a single part (right upper leg and right lower leg become the same class. right leg)
body_parts_dict = {
    'head':1,
    'lear':1,
    'rear':1,
    'mouth':1,
    'hair':1,
    'nose':1,
    'leye':1,
    'reye':1,
    'lebrow':1,
    'rebrow':1,
    'torso':2,
    'neck':2,
    'luarm':3,
    'llarm':3,
    'lhand':3,
    'rlarm':4,
    'ruarm':4,
    'rhand':4,
    'llleg':5,
    'luleg':5,
    'lfoot':5,
    'rlleg':6,
    'ruleg':6,
    'rfoot':6
}

body_parts_dict = {
    'head':1,
    'lear':1,
    'rear':1,
    'mouth':1,
    'hair':1,
    'nose':1,
    'leye':1,
    'reye':1,
    'lebrow':1,
    'rebrow':1,
    'torso':2,
    'neck':2,
    'luarm':3,
    'llarm':3,
    'lhand':3,
    'rlarm':3,
    'ruarm':3,
    'rhand':3,
    'llleg':5,
    'luleg':5,
    'lfoot':5,
    'rlleg':5,
    'ruleg':5,
    'rfoot':5
}
#load data takes some form of annotations (provided from dataset) and return the correct form of annotation for tfrecords
def loadData3(H,W): ####  !!!!! i know i do not send the image and the annotations into the function but python is smart enough to look outside the scope of the function. look below

    masks_instances = []#shape: [N,H,W,7]

    persons = [o for o in annotation['anno'][0]['objects'][0][0] if o['class']=='person'] # select all persons from the image
    gt_boxes = [] #will have shape: [N,x1,y1,x2,y2,cls]
    for i in range(len(persons)):
        p = persons[i]
        pa = p['parts']
        parts = pa[0]
        masks_for_person = np.zeros((H,W,7),dtype=np.uint8) # whole body +  6 parts
        one_mask_person = np.zeros((H,W),dtype=np.uint8) # whole body

        for part in parts:
            part_name = part['part_name'].astype(str)[0]
            index = body_parts_dict[part_name]
            masks_for_person[...,index] = np.logical_or(masks_for_person[...,index], part['mask'])#this is where I combine for example right upper leg and right lower leg
            one_mask_person=np.logical_or(one_mask_person,part['mask'])  # this is where I combine the part mask into the whole body

        masks_for_person[...,0]=one_mask_person
        kernel = np.ones((5,5),np.uint8)
        one_mask_person = np.array(one_mask_person,dtype=np.uint8)#if this line is missing=> error in cv2.dilate
        one_mask_person = cv2.dilate(one_mask_person,kernel,iterations = 1)
        _,contours,hierarchy = cv2.findContours(one_mask_person, 1, 2) #### from here
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
        gt_boxes.append([x1,y1,x2,y2,1])

        if True:#########################Body
            print ("BODYYYYYYYYYYYYYYYY")
            masks_for_person = np.zeros((H,W,7),dtype=np.uint8)
            masks_for_person[...,0]=one_mask_person

        masks_instances.append(masks_for_person.copy())                   #####to here I select the bounding box of the person instance. the mask might be splitted into multiple blobs
    if len(gt_boxes) ==0:
        return False,None,None,None

    masks_instances = np.array(masks_instances,dtype=np.uint8)
    gt_boxes = np.array(gt_boxes,dtype=np.float32)
    mask = masks_instances[0,:,:,1]# this is for drawing the ground truth in the network  in tensorboard
    return True,gt_boxes,masks_instances,mask

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
record_filename = "data/out_human_and_body_parts.tfrecord"
with tf.python_io.TFRecordWriter(record_filename, options=options) as tfrecord_writer:
    for x in range (0,12000):
        try:
            img_id = x
            img_name = '2008_%06d' % (x,)
            img = np.array(Image.open('data/JPEGImages/'+img_name+'.jpg'))
            annotation = sio.loadmat('data/Annotations_Part/'+img_name+'.mat')
            image = cv2.imread('data/JPEGImages/'+img_name+'.jpg')
            height, width = img.shape[0],img.shape[1]
            img = img.astype(np.uint8)
            img_raw = img.tostring()
            persons_exist, gt_boxes, masks,mask = loadData3(height, width)
            if not persons_exist:
                continue
            mask_raw = mask.tostring()

            example = _to_tfexample_coco_raw(
                  img_id,
                  img_raw,
                  mask_raw,
                  height, width, gt_boxes.shape[0],
                  gt_boxes.tostring(), masks.tostring())
            tfrecord_writer.write(example.SerializeToString())
            print (x)
        except BaseException as error:
            print error

    tfrecord_writer.close()
