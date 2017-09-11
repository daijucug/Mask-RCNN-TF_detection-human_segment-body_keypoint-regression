import cv2
import ChalearnLAPSample
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

body_parts_dict={
    1:1,#head
    2:2,#torso
    3:3,#left hand
    5:3,#left forearm (lower)
    7:3,#left upper arm
    4:4,#right hand
    6:4,#right forearm
    8:4,#right upperarm
    9:5,#left foot
    11:5,#left lower leg
    13:5,#left upper leg
    10:6,#right foot
    12:6,#right lower leg
    14:6,#rihgt upper leg
}

# poseSample = ChalearnLAPSample.PoseSample("Seq01.zip")
# actorid=1
# limbid=2
# cv2.namedWindow("Seqxx",cv2.WINDOW_NORMAL)
# cv2.namedWindow("Torso",cv2.WINDOW_NORMAL)
# for x in range(1, poseSample.getNumFrames()):
#     img=poseSample.getRGB(x)
#     torso=poseSample.getLimb(x,actorid,6)
#     cv2.imshow("Seqxx",img)
#     cv2.imshow("Torso",torso)
#     cv2.waitKey(1000)
# cv2.destroyAllWindows()


def loadData(frame_id,img,poseSample):
    H,W = img.shape[0],img.shape[1]
    gt_boxes = []
    masks_instances = []
    for actorid in range(1,3):
        masks_for_person = np.zeros((H,W,7),dtype=np.uint8)
        one_mask_person = np.zeros((H,W),dtype=np.uint8)
        for limbid in range(1,11):
            part = poseSample.getLimb(frame_id,actorid,limbid)
            part = cv2.resize(part[...,0]/255,(W,H))
            masks_for_person[...,body_parts_dict[limbid]] = np.logical_or(masks_for_person[...,body_parts_dict[limbid]],part)
            one_mask_person=np.logical_or(one_mask_person,part)
        _,contours,hierarchy = cv2.findContours(one_mask_person.astype(np.uint8).copy(), 1, 2)
        if len(contours)==0:
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
        masks_instances.append(masks_for_person)
        maximum = masks_for_person.max()
    if len(gt_boxes) ==0:
        return False,None,None,None,H,W
    masks_instances = np.array(masks_instances,dtype=np.uint8)
    gt_boxes = np.array(gt_boxes,dtype=np.float32)
    # for h_box in gt_boxes:
    #     image = cv2.rectangle(img,(h_box[0],h_box[1]),(h_box[2],h_box[3]),(255,255,255),2)
    #     cv2.imshow("img",image)
    #     cv2.waitKey(100)
    mask = masks_instances[0,:,:,1]
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
record_filename = "out_human_and_body_parts_chalearn.tfrecord"
with tf.python_io.TFRecordWriter(record_filename, options=options) as tfrecord_writer:
    for seq,seq_id in zip(["Seq01.zip","Seq02.zip","Seq03.zip","Seq04.zip","Seq06.zip"],range(5)):
    #for seq,seq_id in zip(["Seq01.zip"],range(5)):
        poseSample = ChalearnLAPSample.PoseSample(seq)
        for x in range(1, poseSample.getNumFrames(),6):
            img=poseSample.getRGB(x)
            img_id = seq_id*2000+x
            persons_exist,gt_boxes,masks_instances,mask,H,W = loadData(x,img,poseSample)
            if not persons_exist:
                continue
            mask_raw = mask.tostring()
            img_raw = img.tostring()
            example = _to_tfexample_coco_raw(
                img_id,
                img_raw,
                mask_raw,
                H, W, gt_boxes.shape[0],
                gt_boxes.tostring(), masks_instances.tostring())
            print x

            tfrecord_writer.write(example.SerializeToString())
    tfrecord_writer.close()



