import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
import numpy as np
from PIL import Image
import scipy.io as sio
import cv2

def loadData():
    annotation = sio.loadmat('2008_000652.mat')
    annotation = annotation['anno'][0]['objects'][0]['parts'][0][0]['mask'][0]

    masks = np.zeros((18,213,320),dtype=np.uint8)########################this is float32
    gt_boxes = np.zeros((18,5),dtype=np.float32)
    classes = np.zeros((18,1),dtype=np.float32)
    for i in range(0,18):
        mask = annotation[i]###############################*255
        masks[i,...] = mask
        mask = mask.copy()
        _,contours,hierarchy = cv2.findContours(mask, 1, 2)
        x,y,w,h = cv2.boundingRect(contours[0])
        gt_boxes[i,0]=x
        gt_boxes[i,1]=y
        gt_boxes[i,2]=x+w
        gt_boxes[i,3]=y+h
        gt_boxes[i,4] = i


    mask = masks[i,...]# this is for drawing the ground truth in the network
    return gt_boxes,masks,mask

def loadData2():
    annotation = sio.loadmat('2008_000652.mat')
    annotation = annotation['anno'][0]['objects'][0]['parts'][0][0]['mask'][0]

    masks = np.zeros((6,213,320),dtype=np.uint8)########################this is float32
    gt_boxes = np.zeros((6,5),dtype=np.float32)
    classes = np.zeros((6,1),dtype=np.float32)
    o = 0
    for i in range(0,18):
        mask = annotation[i]###############################*255

        mask = mask.copy()
        _,contours,hierarchy = cv2.findContours(mask, 1, 2)
        x,y,w,h = cv2.boundingRect(contours[0])
        if w<50 or h<50:
            continue

        masks[o,...] = mask
        gt_boxes[o,0]=x
        gt_boxes[o,1]=y
        gt_boxes[o,2]=x+w
        gt_boxes[o,3]=y+h
        gt_boxes[o,4] = i
        o = o+1

    print o


    mask = masks[o,...]# this is for drawing the ground truth in the network
    return gt_boxes,masks,mask

body_parts_dict = {
    'head':0,
    'lear':1,
    'rear':2,
    'mouth':3,
    'hair':4,
    'nose':5,
    'leye':6,
    'reye':7,
    'lebrow':8,
    'rebrow':9,
    'torso':10,
    'neck':11,
    'luarm':12,
    'llarm':13,
    'rlarm':14,
    'ruarm':15,
    'rhand':16
}

def loadData3():#human body parts
    annotation = sio.loadmat('2008_003228.mat')
    #annotation = annotation['anno'][0]['objects'][0]['parts'][0][0]['mask'][0]


    # sa zicem ca am doua clase, dar cea de a doua nu o sa apara niciodata
    masks = np.zeros((17,375,500),dtype=np.uint8)
    gt_boxes = np.zeros((3,5),dtype=np.float32)
    for i in range(4):
        obj = annotation['anno'][0]['objects'][0][0][i]
        if (obj['class']=='person'):
            parts = obj['parts'][0]
            contour_mask = np.zeros((375,500),dtype=np.uint8)
            for part in parts:
                contour_mask = contour_mask | part['mask']
                name = part['part_name'].astype(str)[0]
                m = body_parts_dict[name]
                masks[m,...] = part['mask']
            B = np.argwhere(contour_mask==1)
            (y1, x1), (y2, x2) = B.min(0), B.max(0)
            gt_boxes[i,0]=x1
            gt_boxes[i,1]=y1
            gt_boxes[i,2]=x2
            gt_boxes[i,3]=y2
            gt_boxes[i,4] = 1 # clasa 1 inseamna human

            #contour_mask = cv2.cvtColor(contour_mask*255,cv2.COLOR_GRAY2BGR)
            #contour_mask =cv2.rectangle(contour_mask,(x1, y1), (x2, y2),(255,255,0),2)
            #cv2.imshow("image",contour_mask)
            #cv2.waitKey(3000)

    mask = masks[i,...]# this is for drawing the ground truth in the network
    return gt_boxes,masks,mask

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
record_filename = "out.tfrecord"
with tf.python_io.TFRecordWriter(record_filename, options=options) as tfrecord_writer:
    for x in range (100):
        img_id = x
        img_name = '2008_003228.jpg'
        height, width = 375,500
        img = np.array(Image.open(img_name))
        img = img.astype(np.uint8)
        img_raw = img.tostring()
        persons_exist, gt_boxes, masks,mask = loadData3()
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
    tfrecord_writer.close()
