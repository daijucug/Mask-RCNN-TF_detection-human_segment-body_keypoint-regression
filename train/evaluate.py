#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
import time
import numpy as np
import tensorflow as tf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import libs.nets.nets_factory as network
import libs.nets.pyramid_network as pyramid_network
import libs.nets.resnet_v1 as resnet_v1
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
from learn_to_draw.metric import metric_for_image
from train.train_utils import _configure_learning_rate, _configure_optimizer, \
  _get_variables_to_train, get_var_list_to_restore

FLAGS = tf.app.flags.FLAGS
resnet50 = resnet_v1.resnet_v1_50

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def solve(global_step):
    """add solver to losses"""
    # learning reate
    lr = _configure_learning_rate(82783, global_step)
    optimizer = _configure_optimizer(lr)
    tf.summary.scalar('learning_rate', lr)

    # compute and apply gradient
    losses = tf.get_collection(tf.GraphKeys.LOSSES)
    regular_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regular_loss = tf.add_n(regular_losses)
    out_loss = tf.add_n(losses)
    total_loss = tf.add_n(losses + regular_losses)

    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('out_loss', out_loss)
    tf.summary.scalar('regular_loss', regular_loss)

    update_ops = []
    variables_to_train = _get_variables_to_train()
    # update_op = optimizer.minimize(total_loss)
    gradients = optimizer.compute_gradients(total_loss, var_list=variables_to_train)

    # they separate these operations of the optimizer because they send only a subset of trainable variables
    grad_updates = optimizer.apply_gradients(gradients, 
            global_step=global_step)
    update_ops.append(grad_updates)
    
    # update moving mean and variance
    if FLAGS.update_bn:
        update_bns = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_bn = tf.group(*update_bns)
        update_ops.append(update_bn)

    return tf.group(*update_ops)

def restore(sess):
     """choose which param to restore"""
     if FLAGS.restore_previous_if_exists:
        try:
            #this fragment of code will restore graph variables that are found in checkpoint. if there
            #are variables in the current graph that are not in checkpoint, it will not generate an error
            #the values of those variables will be random initialized. this was useful for me when I initially
            #trained the model for bounding box regression and later added the mask branch. the mask branch is harder to train
            print (FLAGS.train_dir)
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
            reader = tf.train.NewCheckpointReader(checkpoint_path)
            saved_shapes = reader.get_variable_to_shape_map()
            var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()if var.name.split(':')[0] in saved_shapes])
            restore_vars = []
            name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
            with tf.variable_scope('', reuse=True):
                for var_name, saved_var_name in var_names:
                    curr_var = name2var[saved_var_name]
                    var_shape = curr_var.get_shape().as_list()
                    if var_shape == saved_shapes[saved_var_name]:
                        restore_vars.append(curr_var)

            restorer = tf.train.Saver(restore_vars)
            restorer.restore(sess, checkpoint_path)
            print ('restored previous model %s from %s'\
                    %(checkpoint_path, FLAGS.train_dir))
            time.sleep(2)
            return
        except:
            print ('--restore_previous_if_exists is set, but failed to restore in %s %s'\
                    % (FLAGS.train_dir, checkpoint_path))
            time.sleep(2)

     if FLAGS.pretrained_model:
        if tf.gfile.IsDirectory(FLAGS.pretrained_model):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.pretrained_model)
        else:
            checkpoint_path = FLAGS.pretrained_model

        if FLAGS.checkpoint_exclude_scopes is None:
            FLAGS.checkpoint_exclude_scopes='pyramid'
        if FLAGS.checkpoint_include_scopes is None:
            FLAGS.checkpoint_include_scopes='resnet_v1_50'

        vars_to_restore = get_var_list_to_restore()
        for var in vars_to_restore:
            print ('restoring ', var.name)
      
        try:
           restorer = tf.train.Saver(vars_to_restore)
           restorer.restore(sess, checkpoint_path)
           print ('Restored %d(%d) vars from %s' %(
               len(vars_to_restore), len(tf.global_variables()),
               checkpoint_path ))
        except:
           print ('Checking your params %s' %(checkpoint_path))
           raise

def get_placeholders():
    image = tf.placeholder(tf.uint8,shape=[None,None,3])
    gt_boxes = tf.placeholder(tf.float32,shape=[None,5])
    gt_masks = tf.placeholder(tf.int32,shape=[None, None, None,7])
    ih = tf.placeholder(tf.int32,shape=None)
    iw = tf.placeholder(tf.int32,shape=None)
    num_instances = tf.placeholder(tf.int32,shape=None)
    img_id = tf.placeholder(tf.int32,shape=None)
    return image, ih, iw, gt_boxes, gt_masks, num_instances, img_id

#this function reads from tfrecord and returns the inputs and labels
def get_npy_arrays(example):
    f = example.features.feature
    img_idnp = f['image/img_id'].int64_list.value[0]
    image_np = f['image/encoded'].bytes_list.value[0]
    heightnp = f['image/height'].int64_list.value[0]
    widthnp = f['image/width'].int64_list.value[0]
    num_instancesnp = f['label/num_instances'].int64_list.value[0]
    gt_masksnp = f['label/gt_masks'].bytes_list.value[0]
    gt_boxesnp = f['label/gt_boxes'].bytes_list.value[0]
    image_np = np.fromstring(image_np, dtype=np.uint8)
    image_np = image_np.reshape((heightnp, widthnp, 3))
    gt_masksnp = np.fromstring(gt_masksnp, dtype=np.uint8)
    gt_masksnp = gt_masksnp.reshape((num_instancesnp, heightnp, widthnp,7))
    gt_boxesnp = np.fromstring(gt_boxesnp, dtype=np.float32)
    gt_boxesnp = gt_boxesnp.reshape((num_instancesnp,5))
    return image_np,heightnp,widthnp,gt_boxesnp,gt_masksnp,num_instancesnp,img_idnp

def train():

    imageo, ih, iw, gt_boxes, gt_masks, num_instances, img_id = get_placeholders()
    image = tf.cast(imageo, tf.float32)
    image = image / 255.0
    image = (image - 0.5) * 2.0
    image = tf.expand_dims(image, axis=0)
    image = tf.reverse(image, axis=[-1])
    im_shape = tf.shape(image)

    ## network
    logits, end_points, pyramid_map = network.get_network(FLAGS.network, image, weight_decay=FLAGS.weight_decay, is_training=True)
    outputs = pyramid_network.build(end_points, im_shape[1], im_shape[2], pyramid_map, num_classes=2, base_anchors=9, is_training=False, gt_boxes=gt_boxes, gt_masks=gt_masks, loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0])

    input_image = end_points['input']
    final_box = outputs['final_boxes']['box']
    final_cls = outputs['final_boxes']['cls']
    final_prob = outputs['final_boxes']['prob']
    final_gt_cls = outputs['final_boxes']['gt_cls']
    final_mask = outputs['mask']['final_mask_for_drawing']

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    ## restore
    restore(sess)

    metrics = []
    metricsp0 = []
    metricsp1 = []
    metricsp2 = []
    metricsp3 = []
    metricsp4 = []
    metricsp5 = []
    metricsp6 = []

    #training tfrecords
    #tfrecords = ['data/coco/records/out_VOC.tfrecord','data/coco/records/out_ade20K.tfrecord','data/coco/records/out_chalearn.tfrecord','data/coco/records/out_keypoints_JHMDB.tfrecord']
    #test tfrecords
    tfrecords = ['data/coco/records/out_freiburg.tfrecord']
    options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
    example = tf.train.Example()
    counter = 0

    step = 0
    for record_name in tfrecords:
        for record in tf.python_io.tf_record_iterator(record_name,options):
            example.ParseFromString(record)
            imagenp, ihnp, iwnp, gt_boxesnp, gt_masksnp, num_instancesnp, img_idnp = get_npy_arrays(example)
            start_time = time.time()
            img_id_str, gt_boxesnp, input_imagenp, final_boxnp, final_clsnp, final_gt_clsnp, final_probnp,  final_masknp,gt_masksnp= sess.run( [img_id] +  [gt_boxes] + [input_image] + [final_box] + [final_cls] + [final_gt_cls] + [final_prob] +  [final_mask]+[gt_masks],feed_dict={imageo:imagenp, ih:ihnp, iw:iwnp, gt_boxes:gt_boxesnp, gt_masks:gt_masksnp, num_instances:num_instancesnp, img_id:img_idnp })

            final_gt_clsnp = np.argmax(np.asarray(final_gt_clsnp),axis=1)
            metrics.append(metric_for_image(-1,final_boxnp,gt_boxesnp,final_clsnp,final_gt_clsnp,final_probnp,final_masknp,gt_masksnp))
            metricsp0.append(metric_for_image(0,final_boxnp,gt_boxesnp,final_clsnp,final_gt_clsnp,final_probnp,final_masknp,gt_masksnp))
            metricsp1.append(metric_for_image(1,final_boxnp,gt_boxesnp,final_clsnp,final_gt_clsnp,final_probnp,final_masknp,gt_masksnp))
            metricsp2.append(metric_for_image(2,final_boxnp,gt_boxesnp,final_clsnp,final_gt_clsnp,final_probnp,final_masknp,gt_masksnp))
            metricsp3.append(metric_for_image(3,final_boxnp,gt_boxesnp,final_clsnp,final_gt_clsnp,final_probnp,final_masknp,gt_masksnp))
            metricsp4.append(metric_for_image(4,final_boxnp,gt_boxesnp,final_clsnp,final_gt_clsnp,final_probnp,final_masknp,gt_masksnp))
            metricsp5.append(metric_for_image(5,final_boxnp,gt_boxesnp,final_clsnp,final_gt_clsnp,final_probnp,final_masknp,gt_masksnp))
            metricsp6.append(metric_for_image(6,final_boxnp,gt_boxesnp,final_clsnp,final_gt_clsnp,final_probnp,final_masknp,gt_masksnp))
            print (reduce(lambda x, y: x + y, metrics) / len(metrics))
            duration_time = time.time() - start_time
            counter = counter + 1
            if step % 100 == 0:
                if FLAGS.INCLUDE_MASK:
                    print ( """iter %d: image-id:%07d, time:%.3f(sec) """
                            """instances: %d, proposals: %d """
                           % (step, img_id_str, duration_time, gt_boxesnp.shape[0],len(final_boxnp)))

    print ("LAST mAP complete :", reduce(lambda x, y: x + y, metrics) / len(metrics))
    print ("LAST mAP body     :", reduce(lambda x, y: x + y, metricsp0) / len(metricsp0))
    print ("LAST mAP head     :", reduce(lambda x, y: x + y, metricsp1) / len(metricsp1))
    print ("LAST mAP torso    :", reduce(lambda x, y: x + y, metricsp2) / len(metricsp2))
    print ("LAST mAP l arm    :", reduce(lambda x, y: x + y, metricsp3) / len(metricsp3))
    print ("LAST mAP r arm    :", reduce(lambda x, y: x + y, metricsp4) / len(metricsp4))
    print ("LAST mAP l leg    :", reduce(lambda x, y: x + y, metricsp5) / len(metricsp5))
    print ("LAST mAP r leg    :", reduce(lambda x, y: x + y, metricsp6) / len(metricsp6))

if __name__ == '__main__':
    train()
