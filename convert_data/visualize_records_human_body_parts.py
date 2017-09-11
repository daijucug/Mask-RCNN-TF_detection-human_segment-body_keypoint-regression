import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
import numpy as np
import cv2

random_color =np.random.randint(0,180,(7))

example = tf.train.Example()
options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
#for record in tf.python_io.tf_record_iterator('data/out_human_and_keypoints_to_body_parts_COCO.tfrecord',options):
#for record in tf.python_io.tf_record_iterator('data/out_human_and_body_parts_ade_20k_max640edge.tfrecord',options):
#for record in tf.python_io.tf_record_iterator('data/out_human_and_body_parts.tfrecord',options):
#for record in tf.python_io.tf_record_iterator('data/chalearn/api_code/out_human_and_body_parts_chalearn.tfrecord',options):
for record in tf.python_io.tf_record_iterator('data/freiburg/out_human_and_body_parts_Freiburg.tfrecord',options):
#for record in tf.python_io.tf_record_iterator('data/jhmdb/out_human_and_body_parts_keypoints_JHMDB.tfrecord',options):
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
            S = 255
            if mask_part ==0:
                S=100
            for x in range(int(h_box[0]),int(h_box[2])):
                for y in range(int(h_box[1]),int(h_box[3])):
                    if mask[y,x]==1:
                        hsv[y,x,0] = random_color[mask_part]
                        hsv[y,x,1] = S
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow("img",bgr)
    cv2.waitKey(1000)


