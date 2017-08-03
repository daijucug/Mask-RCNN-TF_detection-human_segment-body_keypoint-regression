import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
import numpy as np
from PIL import Image
import scipy.io as sio
import cv2

options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
reader = tf.TFRecordReader(options=options)
filename_queue = tf.train.string_input_producer(
    ['data/out_human_and_body_parts.tfrecord'], num_epochs=100)
_, serialized_example = reader.read(filename_queue)


features = tf.parse_single_example(
    serialized_example,
    features={
      'image/img_id': tf.FixedLenFeature([], tf.int64),
      'image/encoded': tf.FixedLenFeature([], tf.string),
      'image/height': tf.FixedLenFeature([], tf.int64),
      'image/width': tf.FixedLenFeature([], tf.int64),
      'label/num_instances': tf.FixedLenFeature([], tf.int64),
      'label/gt_masks': tf.FixedLenFeature([], tf.string),
      'label/gt_boxes': tf.FixedLenFeature([], tf.string),
      'label/encoded': tf.FixedLenFeature([], tf.string),
      })


img_id = tf.cast(features['image/img_id'], tf.int32)
ih = tf.cast(features['image/height'], tf.int32)
iw = tf.cast(features['image/width'], tf.int32)
num_instances = tf.cast(features['label/num_instances'], tf.int32)
image = tf.decode_raw(features['image/encoded'], tf.uint8)
imsize = tf.size(image)
image = tf.cond(tf.equal(imsize, ih * iw), \
      lambda: tf.image.grayscale_to_rgb(tf.reshape(image, (ih, iw, 1))), \
      lambda: tf.reshape(image, (ih, iw, 3)))

gt_boxes = tf.decode_raw(features['label/gt_boxes'], tf.float32)
gt_boxes = tf.reshape(gt_boxes, [num_instances, 5])
gt_masks = tf.decode_raw(features['label/gt_masks'], tf.uint8)
gt_masks = tf.cast(gt_masks, tf.int32)
gt_masks = tf.reshape(gt_masks, [num_instances, ih, iw,7])
####################################################################be careful here. before 17 at the line above there was num_instances


init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    img_idnp,ihnp,iwnp,num_instancesnp,imagenp,imsizenp,gt_boxesnp,gt_masksnp = sess.run([img_id,ih,iw,num_instances,image,imsize,gt_boxes,gt_masks])
    cv2.imshow("img",imagenp)
    cv2.waitKey(500)
    for h_box in gt_boxesnp:
        image = cv2.rectangle(imagenp,(h_box[0],h_box[1]),(h_box[2],h_box[3]),(255,255,255),2)
    cv2.imshow("img",imagenp)
    cv2.waitKey(500)
    for human_masks in gt_masksnp:
        for mask_part in range(7):
            mask = human_masks[:,:,mask_part]
            mask = mask.astype(np.uint8)
            mask = mask * 255
            cv2.imshow("mask",mask)
            cv2.waitKey(1000)

print ('ok')
#img_idnp is int32 but in tf record is float32
#gt_masks is int32 and also in tfecord is int32 (N,H,W)
#image is uint8 in both forms
#num instances is int32

