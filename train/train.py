#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import functools
import os, sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import gmtime, strftime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import libs.datasets.dataset_factory as datasets
import libs.nets.nets_factory as network
import libs.nets.pyramid_network as pyramid_network
import libs.nets.resnet_v1 as resnet_v1
import sys

from train.train_utils import _configure_learning_rate, _configure_optimizer, \
  _get_variables_to_train, get_var_list_to_restore

from draw.utils import draw_human_body_parts
FLAGS = tf.app.flags.FLAGS
resnet50 = resnet_v1.resnet_v1_50

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

SAVE=False

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

def save(step,input_imagenp,final_boxnp,gt_boxesnp,final_clsnp,final_probnp,final_gt_clsnp,final_masknp=None,gt_masksnp=None):
    save_array = []
    save_array.append(np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0))
    save_array.append(final_boxnp)
    save_array.append(final_clsnp)
    save_array.append(final_probnp)
    save_array.append(gt_boxesnp)
    save_array.append(np.argmax(np.asarray(final_gt_clsnp),axis=1))
    save_array.append(final_masknp)
    save_array.append(gt_masksnp)
    save_array = np.asarray(save_array)
    np.save("array"+str(step)+".npy",save_array)

def train():
    """The main function that runs training"""
    ## data
    #this will return the placeholders from tfrecords
    image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = datasets.get_dataset(FLAGS.dataset_name,  FLAGS.dataset_split_name, FLAGS.dataset_dir, FLAGS.im_batch,is_training=True)

    data_queue = tf.RandomShuffleQueue(capacity=32, min_after_dequeue=16,dtypes=(
                image.dtype, ih.dtype, iw.dtype, 
                gt_boxes.dtype, gt_masks.dtype, 
                num_instances.dtype, img_id.dtype)) 
    enqueue_op = data_queue.enqueue((image, ih, iw, gt_boxes, gt_masks, num_instances, img_id))
    data_queue_runner = tf.train.QueueRunner(data_queue, [enqueue_op] * 4)
    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, data_queue_runner)
    (image, ih, iw, gt_boxes, gt_masks, num_instances, img_id) =  data_queue.dequeue()
    im_shape = tf.shape(image)
    image = tf.reshape(image, (im_shape[0], im_shape[1], im_shape[2], 3))

    ## network
    logits, end_points, pyramid_map = network.get_network(FLAGS.network, image,weight_decay=FLAGS.weight_decay, is_training=True)
    outputs = pyramid_network.build(end_points, im_shape[1], im_shape[2], pyramid_map,num_classes=2, base_anchors=9,is_training=True, gt_boxes=gt_boxes, gt_masks=gt_masks,loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0])


    total_loss = outputs['total_loss']
    losses  = outputs['losses']

    regular_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    
    input_image = end_points['input']
    final_box = outputs['final_boxes']['box']
    final_cls = outputs['final_boxes']['cls']
    final_prob = outputs['final_boxes']['prob']
    final_gt_cls = outputs['final_boxes']['gt_cls']

    #this flag is used for including the mask or not. initally I trained the network without the mask branch, because I wanted to train better the region proposal network
    # so that the network proposes better boxes. If the boxes are better proposed, the branch network will learn easier. Initially I thought that this is the problem
    # for the model memory issue. The idea is that at some point the network was proposing too many regions, like 120, and the Tensor for the mask branch would cause an out of memory error
    # because the shape of tensor would be [120,112,112,7]
    print ("FLAGS INCLUDE MASK IS ",FLAGS.INCLUDE_MASK)
    if FLAGS.INCLUDE_MASK:
        final_mask = outputs['mask']['final_mask_for_drawing']
    gt = outputs['gt']

    

    #############################
    tmp_0 = outputs['losses']
    tmp_1 = outputs['losses']
    tmp_2 = outputs['losses']
    tmp_3 = outputs['tmp_3']
    tmp_4 = outputs['tmp_4']
    ############################


    ## solvers
    global_step = slim.create_global_step()
    update_op = solve(global_step)
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
            )
    sess.run(init_op)

    summary_op = tf.summary.merge_all()
    logdir = os.path.join(FLAGS.train_dir, strftime('%Y%m%d%H%M%S', gmtime()))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

    ## restore
    restore(sess)

    ## main loop
    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

    tf.train.start_queue_runners(sess=sess, coord=coord)
    saver = tf.train.Saver(max_to_keep=20)

    for step in range(FLAGS.max_iters):
        
        start_time = time.time()
        if FLAGS.INCLUDE_MASK:
            s_, tot_loss, reg_lossnp, img_id_str, rpn_box_loss, rpn_cls_loss, refined_box_loss, refined_cls_loss,mask_loss, gt_boxesnp, input_imagenp, final_boxnp, final_clsnp, final_probnp, final_gt_clsnp, gtnp, tmp_0np, tmp_1np, tmp_2np, tmp_3np, tmp_4np, final_masknp,gt_masksnp= sess.run([update_op, total_loss, regular_loss, img_id] + losses + [gt_boxes] + [input_image] + [final_box] + [final_cls] + [final_prob] + [final_gt_cls] + [gt] + [tmp_0] + [tmp_1] + [tmp_2] + [tmp_3] + [tmp_4]+[final_mask]+[gt_masks])
        else:
            s_, tot_loss, reg_lossnp, img_id_str,\
            rpn_box_loss, rpn_cls_loss,refined_box_loss, refined_cls_loss,\
            gt_boxesnp, input_imagenp, final_boxnp,\
            final_clsnp, final_probnp, final_gt_clsnp, gtnp=\
                sess.run([update_op, total_loss, regular_loss, img_id] +\
                         losses +\
                         [gt_boxes] + [input_image] + [final_box] + \
                         [final_cls] + [final_prob] + [final_gt_cls] + [gt])

        duration_time = time.time() - start_time
        if step % 1 == 0:
            if FLAGS.INCLUDE_MASK:
                print ( """iter %d: image-id:%07d, time:%.3f(sec), regular_loss: %.9f, """
                        """total-loss %.10f(%.4f, %.4f, %.6f, %.4f,%.5f), """ #%.4f
                        """instances: %d, proposals: %d """
                       % (step, img_id_str, duration_time, reg_lossnp,
                          tot_loss, rpn_box_loss, rpn_cls_loss, refined_box_loss, refined_cls_loss, mask_loss,
                          gt_boxesnp.shape[0],len(final_boxnp)))
            else:
                print ( """iter %d: image-id:%07d, time:%.3f(sec), regular_loss: %.9f, """
                        """total-loss %.4f(%.4f, %.4f, %.6f, %.4f), """ #%.4f
                        """instances: %d, proposals: %d """
                       % (step, img_id_str, duration_time, reg_lossnp,
                          tot_loss, rpn_box_loss, rpn_cls_loss, refined_box_loss, refined_cls_loss, #mask_loss,
                          gt_boxesnp.shape[0],len(final_boxnp)))

            if sys.argv[1]=='--draw':
                if FLAGS.INCLUDE_MASK:
                    input_imagenp = np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0)
                    final_gt_clsnp = np.argmax(np.asarray(final_gt_clsnp),axis=1)
                    draw_human_body_parts(step, input_imagenp,  bbox=final_boxnp, label=final_clsnp, gt_label=final_gt_clsnp, prob=final_probnp,final_mask=final_masknp)

                else:
                    save(step,input_imagenp,final_boxnp,gt_boxesnp,final_clsnp,final_probnp,final_gt_clsnp,None,None)

            if np.isnan(tot_loss) or np.isinf(tot_loss):
                print (gt_boxesnp)
                raise
          
        if step % 1000 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        if (step % 1000 == 0 or step + 1 == FLAGS.max_iters) and step != 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 
                                           FLAGS.dataset_name + '_' + FLAGS.network + '_model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

        if coord.should_stop():
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    train()
