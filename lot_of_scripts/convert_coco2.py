#from pycocotools.coco import COCO
from convert_data.data.coco.coco.PythonAPI.pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType


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

body_parts_dict = {
    (15, 13): 5,  # lower left leg
    (13, 11): 5,  # upper left leg
    (16, 14): 6,  # lower right leg
    (14, 12): 6,  # upper right leg
    (11, 12): 2,  # bottom torso
    (5, 11): 2,  # left torso
    (6, 12): 2,  # right torso
    (5, 6): 2,  # upper torso
    (5, 7): 3,  # left upper arm
    (6, 8): 4,  # right upper arm
    (7, 9): 3,  # left lower arm
    (8, 10): 4,  # right lower arm
    (3, 5): 1,  # head
    (4, 6): 1,  # head
    (2, 4): 1,  # head
    (1, 2): 1,  # head
    (1, 3): 1,  # head
    (0, 1): 1,  # head
    (0, 2): 1  # head
}

def load_data(anns,I,coco_kps,H,W):
    gt_boxes = []
    mask_instances = []
    for ann in anns:
        if type(ann['segmentation']) == list:
            # polygon
            person_mask = np.zeros((H,W),dtype=np.uint8)
            for seg in ann['segmentation']:
                poly = np.array(seg,dtype=np.int32).reshape((int(len(seg)/2), 2))
                pts = poly.reshape((-1,1,2))
                person_mask = cv2.fillPoly(person_mask,[pts],(1))
                person_mask = cv2.resize(person_mask,(W,H))

                _,contours,hierarchy = cv2.findContours(person_mask.copy(), 1, 2)
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
                Xb,Yb,Wb,Hb = x1,y1,x2-x1,y2-y1

                size = int((Wb-40)*(24-7)/(200-40)+7)

        masks_for_person = np.zeros((H,W,7),dtype=np.uint8)
        if 'keypoints' in ann and type(ann['keypoints']) == list:
            sks = np.array(coco_kps.loadCats(ann['category_id'])[0]['skeleton'])-1
            kp = np.array(ann['keypoints'])
            x = kp[0::3]
            y = kp[1::3]
            v = kp[2::3]
            torso_list = []
            for sk in sks:
                if np.all(v[sk]>0):
                    #plt.plot(x[sk],y[sk], linewidth=12, color=col)
                    xs = x[sk]
                    ys = y[sk]
                    sk = (sk[0],sk[1])
                    mask = np.zeros((I.shape[0],I.shape[1]),dtype=np.uint8)
                    mask = cv2.line(mask,(xs[0],ys[0]),(xs[1],ys[1]),(1),size)
                    if body_parts_dict[sk]==2:
                        torso_list.append([xs[0],ys[0]])
                        torso_list.append([xs[1],ys[1]])
                    mask = cv2.resize(mask,(W,H))
                    masks_for_person[...,body_parts_dict[sk]] = np.logical_or(masks_for_person[...,body_parts_dict[sk]],mask)
            if len(torso_list)==0:
                continue
            torso_mask = np.zeros((I.shape[0],I.shape[1]),dtype=np.uint8)
            torso_list = np.array(torso_list,dtype=np.int32)
            torso_list = torso_list.reshape((-1,1,2))
            torso_mask = cv2.fillPoly(torso_mask,[torso_list],(1))
            torso_mask = cv2.resize(torso_mask,(W,H))
            masks_for_person[...,2] = np.logical_or(masks_for_person[...,2],torso_mask)
            gt_boxes.append([Xb,Yb,Xb+Wb,Yb+Hb,1])
            # for x in range(7):
            #     cv2.imshow("mask",masks_for_person[...,x]*255)
            #     cv2.waitKey(500)
            mask_instances.append(masks_for_person)

    if len(mask_instances)==0:
        return False,None,None,None
    mask_instances = np.array(mask_instances,dtype=np.uint8)
    gt_boxes = np.array(gt_boxes,dtype=np.float32)
    mask = mask_instances[0,:,:,1]
    return True,gt_boxes,mask_instances,mask


dataDir='..'
dataType='train2014'
annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)
annFile = '%s/annotations/person_keypoints_%s.json'%(dataDir,dataType)
coco_kps=COCO(annFile)
coco=COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds )
options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
record_filename = "/home/alex/PycharmProjects/MaskRCNN_body/convert_data/data/out_human_and_keypoints_to_body_parts_COCO.tfrecord"


with tf.python_io.TFRecordWriter(record_filename, options=options) as tfrecord_writer:
    print (len(imgIds))
    for x in range(1,20):
        img = coco.loadImgs(imgIds[x])[0]
        I = io.imread('http://mscoco.org/images/%d'%(img['id']))
        annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco_kps.loadAnns(annIds)
        h,w = I.shape[0],I.shape[1]
        ratio = float(h)/float(w)
        if h >=w:
            if h>640:
                h=640
                w=int(h/ratio)
        elif w>h:
            if w >640:
                w=640
                h=int(ratio*w)
        I = cv2.resize(I,(w,h))
        I = cv2.cvtColor(I,cv2.COLOR_RGB2BGR)
        Exists_Mask,gt_boxes,masks_instances,mask = load_data(anns,I,coco_kps,h,w)

        if not Exists_Mask:
            continue
        mask_raw = mask.tostring()
        img_raw = I.tostring()
        example = _to_tfexample_coco_raw(
              img['id'],
              img_raw,
              mask_raw,
              h, w, gt_boxes.shape[0],
              gt_boxes.tostring(), masks_instances.tostring())
        tfrecord_writer.write(example.SerializeToString())
        print x
    tfrecord_writer.close()



