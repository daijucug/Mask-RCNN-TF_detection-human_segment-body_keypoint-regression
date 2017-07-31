import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
import numpy as np
from PIL import Image
import scipy.io as sio
import cv2
import traceback
import logging

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

def loadData3(H,W):#human body parts

    #annotation = annotation['anno'][0]['objects'][0]['parts'][0][0]['mask'][0]


    # sa zicem ca am doua clase, dar cea de a doua nu o sa apara niciodata
    #masks_instances = np.zeros((6,H,W),dtype=np.uint8)
    masks_instances = []
    masks_for_person = np.zeros((6,H,W),dtype=np.uint8)
    persons = [o for o in annotation['anno'][0]['objects'][0][0] if o['class']=='person']
    gt_boxes = []
    for i in range(len(persons)):
        p = persons[i]
        pa = p['parts']
        parts = pa[0]
        #parts = persons[i]['parts'][0]
        for part in parts:
            part_name = part['part_name'].astype(str)[0]
            index = body_parts_dict[part_name] -1
            masks_for_person[index,...] = np.logical_or(masks_for_person[index,...], part['mask'])
        for j in range(6):
            mask = masks_for_person[j,...].copy()
            #cv2.imshow("mask",mask*255)
            #cv2.waitKey(100)
            kernel = np.ones((5,5),np.uint8)
            mask = cv2.dilate(mask,kernel,iterations = 2)
            #cv2.imshow("mask",mask*255)
            #cv2.waitKey(1000)
            _,contours,hierarchy = cv2.findContours(mask, 1, 2)
            if len(contours) ==0:
                continue
            x,y,w,h = cv2.boundingRect(contours[0])
            mask = cv2.cvtColor(mask*255,cv2.COLOR_GRAY2BGR)
            #mask = cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,0),2)
            #cv2.imshow("mask",mask)
            #cv2.waitKey(100)
            gt_boxes.append([x,y,x+w,y+h,j])
            masks_instances.append(masks_for_person[j,...].copy())

    if len(gt_boxes) ==0:
        return False,None,None,None

    masks_instances = np.array(masks_instances,dtype=np.uint8)
    gt_boxes = np.array(gt_boxes,dtype=np.float32)


    mask = masks_instances[0,...]# this is for drawing the ground truth in the network
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

annotation = sio.loadmat('/home/alex/PycharmProjects/MaskRCNN_body/convert_data/data/Annotations_Part/2008_000002.mat')

options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
record_filename = "out_5_body_parts.tfrecord"
with tf.python_io.TFRecordWriter(record_filename, options=options) as tfrecord_writer:
    for x in range (930,6086):
        try:
            img_id = x
            img_name = '2008_%06d' % (x,)
            img = np.array(Image.open('/home/alex/PycharmProjects/MaskRCNN_body/convert_data/data/JPEGImages/'+img_name+'.jpg'))
            annotation = sio.loadmat('data/Annotations_Part/'+img_name+'.mat')

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
        except BaseException as error:
            #logging.error(traceback.format_exc())
            print error
            #print img_name+' annotation does not exit'

        # image = cv2.imread(img_name)
        # for x in range(gt_boxes.shape[0]):
        #     c = np.random.randint(0,255,(3))
        #     image = cv2.rectangle(image,(gt_boxes[x,0],gt_boxes[x,1]),(gt_boxes[x,2],gt_boxes[x,3]),(c[0],c[1],c[2]),2)
        # cv2.imshow("iamge",image)
        # cv2.waitKey(3000)
    tfrecord_writer.close()
