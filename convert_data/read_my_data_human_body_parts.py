import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
import numpy as np
from PIL import Image
import scipy.io as sio
import cv2

# options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
# reader = tf.TFRecordReader(options=options)
# # filename_queue = tf.train.string_input_producer(
# #     ['data/out_human_and_body_parts.tfrecord'], num_epochs=100)
# # filename_queue = tf.train.string_input_producer(
# #     ['data/chalearn/api_code/out_human_and_body_parts_chalearn.tfrecord'], num_epochs=1)
# filename_queue = tf.train.string_input_producer(
#     ['data/out_human_and_body_parts_ade_20k_max640edge.tfrecord'], num_epochs=1)
# _, serialized_example = reader.read(filename_queue)
#
#
# features = tf.parse_single_example(
#     serialized_example,
#     features={
#       'image/img_id': tf.FixedLenFeature([], tf.int64),
#       'image/encoded': tf.FixedLenFeature([], tf.string),
#       'image/height': tf.FixedLenFeature([], tf.int64),
#       'image/width': tf.FixedLenFeature([], tf.int64),
#       'label/num_instances': tf.FixedLenFeature([], tf.int64),
#       'label/gt_masks': tf.FixedLenFeature([], tf.string),
#       'label/gt_boxes': tf.FixedLenFeature([], tf.string),
#       'label/encoded': tf.FixedLenFeature([], tf.string),
#       })
#
#
# img_id = tf.cast(features['image/img_id'], tf.int32)
# ih = tf.cast(features['image/height'], tf.int32)
# iw = tf.cast(features['image/width'], tf.int32)
# num_instances = tf.cast(features['label/num_instances'], tf.int32)
# image = tf.decode_raw(features['image/encoded'], tf.uint8)
# imsize = tf.size(image)
# image = tf.cond(tf.equal(imsize, ih * iw), \
#       lambda: tf.image.grayscale_to_rgb(tf.reshape(image, (ih, iw, 1))), \
#       lambda: tf.reshape(image, (ih, iw, 3)))
#
# gt_boxes = tf.decode_raw(features['label/gt_boxes'], tf.float32)
# gt_boxes = tf.reshape(gt_boxes, [num_instances, 5])
# gt_masks = tf.decode_raw(features['label/gt_masks'], tf.uint8)
# gt_masks = tf.cast(gt_masks, tf.int32)
# gt_masks = tf.reshape(gt_masks, [num_instances, ih, iw,7])
# ####################################################################be careful here. before 17 at the line above there was num_instances
#
#
# init_op = tf.group(tf.global_variables_initializer(),
#                    tf.local_variables_initializer())
# with tf.Session() as sess:
#     sess.run(init_op)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#
#     img_idnp,ihnp,iwnp,num_instancesnp,imagenp,imsizenp,gt_boxesnp,gt_masksnp = sess.run([img_id,ih,iw,num_instances,image,imsize,gt_boxes,gt_masks])
#     cv2.imshow("img",imagenp)
#     cv2.waitKey(500)
#     for h_box in gt_boxesnp:
#         image = cv2.rectangle(imagenp,(h_box[0],h_box[1]),(h_box[2],h_box[3]),(255,255,255),2)
#         cv2.imshow("img",image)
#         cv2.waitKey(10)
#     cv2.imshow("img",imagenp)
#     cv2.waitKey(500)
#     for human_masks in gt_masksnp:
#         for mask_part in range(7):
#             mask = human_masks[:,:,mask_part]
#             mask = mask.astype(np.uint8)
#             mask = mask * 255
#             cv2.imshow("mask",mask)
#             cv2.waitKey(30)

colors = []
colors.append([180,255,255])
colors.append([150,255,255])
colors.append([120,255,255])
colors.append([90,255,255])
colors.append([60,255,255])
colors.append([30,255,255])
colors.append([0,255,255])

random_color =np.random.randint(0,180,(7))
i=0
example = tf.train.Example()
options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
#for record in tf.python_io.tf_record_iterator('data/out_human_and_keypoints_to_body_parts_COCO.tfrecord',options):
#for record in tf.python_io.tf_record_iterator('data/out_human_and_body_parts_ade_20k_max640edge.tfrecord',options):
#for record in tf.python_io.tf_record_iterator('data/out_human_and_body_parts.tfrecord',options):

#for record in tf.python_io.tf_record_iterator('data/chalearn/api_code/out_human_and_body_parts_chalearn.tfrecord',options):
#for record in tf.python_io.tf_record_iterator('data/jhmdb/out_human_and_body_parts_keypoints_JHMDB.tfrecord',options):
for record in tf.python_io.tf_record_iterator('data/freiburg/out_human_and_body_parts_Freiburg.tfrecord',options):
    example.ParseFromString(record)
    f = example.features.feature
    img_idnp = f['image/img_id'].int64_list.value[0]
    image_np = f['image/encoded'].bytes_list.value[0]
    heightnp = f['image/height'].int64_list.value[0]
    widthnp = f['image/width'].int64_list.value[0]
    num_instancesnp = f['label/num_instances'].int64_list.value[0]
    gt_masksnp = f['label/gt_masks'].bytes_list.value[0]
    gt_boxesnp = f['label/gt_boxes'].bytes_list.value[0]
    encoded = f['label/encoded'].bytes_list.value[0]
    i = i+1
    print i
    image_np = np.fromstring(image_np, dtype=np.uint8)
    image_np = image_np.reshape((heightnp, widthnp, 3))
    gt_masksnp = np.fromstring(gt_masksnp, dtype=np.uint8)
    gt_masksnp = gt_masksnp.reshape((num_instancesnp, heightnp, widthnp,7))
    gt_boxesnp = np.fromstring(gt_boxesnp, dtype=np.float32)
    gt_boxesnp = gt_boxesnp.reshape((num_instancesnp,5))
    cv2.imshow("img",image_np)
    cv2.waitKey(100)
    hsv = cv2.cvtColor(image_np,cv2.COLOR_BGR2HSV)
    for h_box,human_masks in zip(gt_boxesnp,gt_masksnp):
        hsv = cv2.rectangle(hsv,(h_box[0],h_box[1]),(h_box[2],h_box[3]),(255,255,255),2)
        for mask_part in range(7):
            mask = human_masks[:,:,mask_part]
            mask = mask.astype(np.uint8)

            for x in range(int(h_box[0]),int(h_box[2])):
                for y in range(int(h_box[1]),int(h_box[3])):
                    if mask[y,x]==1:
                        hsv[y,x,0] = colors[mask_part][0]
                        hsv[y,x,1] = 255
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow("img",bgr)
    cv2.waitKey(1000)


