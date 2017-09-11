from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from libs.visualization.summary_utils import visualize_input
import glob
from libs.datasets import coco

import libs.preprocessings.coco_v1 as coco_preprocess

def get_dataset(dataset_name, split_name, dataset_dir, 
        im_batch=1, is_training=False, file_pattern=None, reader=None):
    """"""
    if file_pattern is None:
        file_pattern = '*keypoints_JHMDB.tfrecord'

    tfrecords = glob.glob(dataset_dir + '/records/' + file_pattern)
    print (tfrecords)
    image, ih, iw, gt_boxes, gt_masks, num_instances, img_id,gt_keypoints = coco.read(tfrecords)
    image = tf.cast(image, tf.float32)
    image = image / 256.0
    image = (image - 0.5) * 2.0
    image = tf.expand_dims(image, axis=0)

    return image, ih, iw, gt_boxes, gt_masks, num_instances, img_id,gt_keypoints

