import os
import scipy.io as sio
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType


#the body_parts_dict is used for combining multiple labels into a single part (right upper leg and right lower leg become the same class. right leg)
body_parts_dict = {
    2:1,#head
    1:2,#torso
    4:3,#left upper arm
    8:3,#left lower arm
    3:4,#right upper arm
    7:4,#right lower arm
    6:5,#left upper leg
    5:6,#right upper leg
    9:6,#right lower leg
    10:5,#left lower leg

}

body_parts_dict = {
    2:1,#head
    1:2,#torso
    4:3,#left upper arm
    8:3,#left lower arm
    3:3,#right upper arm
    7:3,#right lower arm
    6:5,#left upper leg
    5:5,#right upper leg
    9:5,#right lower leg
    10:5,#left lower leg
}

# this is used to normalize the x,y of keypoint to -1 and 1
def map_value(x,A,B,a,b):
    return (x-A)*(b-a)/(B-A)+a


#load data takes some form of annotations (provided from dataset) and return the correct form of annotation for tfrecords
def loadData(image,instance_mask,parts_mask,keypoints):
    gt_boxes = [] #will have shape: [N,x1,y1,x2,y2,cls]
    masks_instances = [] #shape: [N,H,W,7]
    _,contours,hierarchy = cv2.findContours(instance_mask.copy(), 1, 2) ######### from here
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
    gt_boxes.append([x1,y1,x2,y2,1])                                     ######### to here i find the bbox of the person as the mask for the person might contain multiple blobs
    H = image.shape[0]
    W = image.shape[1]
    masks_for_person = np.zeros((H,W,7),dtype=np.uint8) # whole body + 6 parts
    masks_for_person[...,0] = instance_mask.copy()
    for x in range(1,11):
        part = (parts_mask == x).astype(np.uint8)
        masks_for_person[...,body_parts_dict[x]] = np.logical_or(masks_for_person[...,body_parts_dict[x]],part) #this is where I combine for example right upper leg and right lower leg

    for x in range(15): #there are 15 keypoints
        # keypoints[0,x] = keypoints[0,x]-x1
        # keypoints[1,x] = keypoints[1,x]-y1
        keypoints[0,x] = map_value(keypoints[0,x],x1,x2,0.0,112.0) #I first normalize to the keypoint to the size of the output mask (112x112) because the keypoint regression branch comes from the mask branch (this is how i decided to attach it)
        keypoints[1,x] = map_value(keypoints[1,x],y1,y2,0.0,112.0)
        keypoints[0,x] = map_value(keypoints[0,x],0.0,112.0,-1,1) #then I normalize it to -1 1 #the above operations are redundant but i left them there for visualization/debugging
        keypoints[1,x] = map_value(keypoints[1,x],0.0,112.0,-1,1)

    if True:####################BODYYY
        masks_for_person = np.zeros((H,W,7),dtype=np.uint8) # whole body + 6 parts
        masks_for_person[...,0] = instance_mask.copy()

    masks_instances.append(masks_for_person)
    masks_instances = np.array(masks_instances,dtype=np.uint8)
    gt_boxes = np.array(gt_boxes,dtype=np.float32)
    mask = masks_instances[0,:,:,1] # this mask is used for visualization in tensorboard
    keypoints = keypoints.astype(np.float32)
    return gt_boxes,masks_instances,mask,H,W,keypoints


def _int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def _to_tfexample_coco_raw(image_id, image_data, label_data,
                           height, width,
                           num_instances, gt_boxes, masks,keypoints):
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
    'label/keypoints': _bytes_feature(keypoints)
  }))

img_id = 0
scenes = os.listdir('JHMDB_video/ReCompress_Videos')
options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
record_filename = "out_human_and_body_parts_keypoints_JHMDB.tfrecord"
with tf.python_io.TFRecordWriter(record_filename, options=options) as tfrecord_writer:
    for s in scenes:
        if s=='.DS_Store':
            continue
        mask_dir = os.listdir('puppet_mask/'+s)
        for mask in mask_dir:
            mat_file_instance = sio.loadmat('puppet_mask/'+s+'/'+mask+'/puppet_mask.mat')
            video_file = cv2.VideoCapture('JHMDB_video/ReCompress_Videos/'+s+'/'+mask+".avi")
            mat_file_parts = sio.loadmat('puppet_flow_com/'+s+'/'+mask+'/puppet_flow.mat')
            mat_file_keypoints = sio.loadmat('joint_positions/'+s+'/'+mask+'/joint_positions.mat')

            #ret, image = video_file.read()
            for x in range(0,mat_file_parts['part_mask'].shape[2]):
                ret, image = video_file.read()
                parts = mat_file_parts['part_mask'][...,x]
                instance = mat_file_instance['part_mask'][...,x]
                keypoints = mat_file_keypoints['pos_img'][...,x]
                # parts = mat_file_parts['part_mask'][...,0]
                # instance = mat_file_instance['part_mask'][...,0]
                # keypoints = mat_file_keypoints['pos_img'][...,0]

                gt_boxes,masks_instances,mask,H,W,keypoints = loadData(image,instance,parts,keypoints)
                mask_raw = mask.tostring()
                img_raw = image.tostring()
                example = _to_tfexample_coco_raw(
                      img_id,
                      img_raw,
                      mask_raw,
                      H, W, gt_boxes.shape[0],
                      gt_boxes.tostring(), masks_instances.tostring(),keypoints.tostring())
                tfrecord_writer.write(example.SerializeToString())

                # cv2.imshow("ar",parts*25)
                # cv2.imshow("image",image)
                # cv2.imshow("instance",instance*255)
                # cv2.waitKey(100)
    tfrecord_writer.close()


