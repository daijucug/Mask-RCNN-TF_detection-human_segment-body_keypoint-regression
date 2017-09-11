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
import socket
import struct
import cv2
import numpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import libs.configs.config_v1 as cfg
import libs.datasets.dataset_factory as datasets
import libs.nets.nets_factory as network 

import libs.preprocessings.coco_v1 as coco_preprocess
import libs.nets.pyramid_network as pyramid_network
import libs.nets.resnet_v1 as resnet_v1
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

from train.train_utils import _configure_learning_rate, _configure_optimizer, \
  _get_variables_to_train, _get_init_fn, get_var_list_to_restore

from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from libs.datasets import download_and_convert_coco
#from libs.datasets.download_and_convert_coco import _cat_id_to_cls_name
from libs.visualization.pil_utils import cat_id_to_cls_name, draw_img, draw_bbox,draw_bbox_better#,draw_segmentation

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
            print (os.path.join(os.getcwd(), FLAGS.train_dir))
            checkpoint_path = tf.train.latest_checkpoint(os.path.join(os.getcwd(), FLAGS.train_dir))
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
            #saver.restore(session, save_file)
            restorer = tf.train.Saver(restore_vars)
            ###########
            # restorer = tf.train.Saver()
            ###########

            ###########
            # not_restore = [ 'pyramid/fully_connected/weights:0', 
            #                 'pyramid/fully_connected/biases:0',
            #                 'pyramid/fully_connected/weights:0', 
            #                 'pyramid/fully_connected_1/biases:0',
            #                 'pyramid/fully_connected_1/weights:0', 
            #                 'pyramid/fully_connected_2/weights:0', 
            #                 'pyramid/fully_connected_2/biases:0',
            #                 'pyramid/fully_connected_3/weights:0', 
            #                 'pyramid/fully_connected_3/biases:0',
            #                 'pyramid/Conv/weights:0', 
            #                 'pyramid/Conv/biases:0',
            #                 'pyramid/Conv_1/weights:0', 
            #                 'pyramid/Conv_1/biases:0', 
            #                 'pyramid/Conv_2/weights:0', 
            #                 'pyramid/Conv_2/biases:0', 
            #                 'pyramid/Conv_3/weights:0', 
            #                 'pyramid/Conv_3/biases:0',
            #                 'pyramid/Conv2d_transpose/weights:0', 
            #                 'pyramid/Conv2d_transpose/biases:0', 
            #                 'pyramid/Conv_4/weights:0',
            #                 'pyramid/Conv_4/biases:0',
            #                 'pyramid/fully_connected/weights/Momentum:0', 
            #                 'pyramid/fully_connected/biases/Momentum:0',
            #                 'pyramid/fully_connected/weights/Momentum:0', 
            #                 'pyramid/fully_connected_1/biases/Momentum:0',
            #                 'pyramid/fully_connected_1/weights/Momentum:0', 
            #                 'pyramid/fully_connected_2/weights/Momentum:0', 
            #                 'pyramid/fully_connected_2/biases/Momentum:0',
            #                 'pyramid/fully_connected_3/weights/Momentum:0', 
            #                 'pyramid/fully_connected_3/biases/Momentum:0',
            #                 'pyramid/Conv/weights/Momentum:0', 
            #                 'pyramid/Conv/biases/Momentum:0',
            #                 'pyramid/Conv_1/weights/Momentum:0', 
            #                 'pyramid/Conv_1/biases/Momentum:0', 
            #                 'pyramid/Conv_2/weights/Momentum:0', 
            #                 'pyramid/Conv_2/biases/Momentum:0', 
            #                 'pyramid/Conv_3/weights/Momentum:0', 
            #                 'pyramid/Conv_3/biases/Momentum:0',
            #                 'pyramid/Conv2d_transpose/weights/Momentum:0', 
            #                 'pyramid/Conv2d_transpose/biases/Momentum:0', 
            #                 'pyramid/Conv_4/weights/Momentum:0',
            #                 'pyramid/Conv_4/biases/Momentum:0',]
            # vars_to_restore = [v for v in  tf.all_variables()if v.name not in not_restore]
            # restorer = tf.train.Saver(vars_to_restore)
            # for var in vars_to_restore:
            #     print ('restoring ', var.name)
            ############

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

def get_placeholders():
    image = tf.placeholder(tf.float32,shape=[None,None,3])
    gt_boxes = tf.placeholder(tf.float32,shape=[None,5])
    gt_masks = tf.placeholder(tf.int32,shape=[None, None, None,7])
    ih = tf.placeholder(tf.int32,shape=None)
    iw = tf.placeholder(tf.int32,shape=None)
    num_instances = tf.placeholder(tf.int32,shape=None)
    img_id = tf.placeholder(tf.int32,shape=None)
    return image, ih, iw, gt_boxes, gt_masks, num_instances, img_id

def get_npy_arrays(example):
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
    return image_np,heightnp,widthnp,gt_boxesnp,gt_masksnp,num_instancesnp,img_idnp

import cv2
import struct
import socket
import numpy as np
import cv2

def getImageFromSocket(s):
    imageSize = s.recv(4)
    imageSize = imageSize[::-1]
    imageSize = struct.unpack('i', imageSize)[0]
    imageBytes = b''
    while imageSize > 0:
        chunk = s.recv(imageSize)
        imageBytes += chunk
        imageSize -= len(chunk)

    data = np.fromstring(imageBytes, dtype='uint8')
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gt_boxesnp = np.zeros((1,5))
    ihnp,iwnp = image.shape[0],image.shape[1]
    img_idnp = 0
    image = np.array(image,np.uint8)
    return image,gt_boxesnp,ihnp,iwnp,img_idnp

def sendImageThroughSocket(socket,image):
    img_str = cv2.imencode('.jpg', image)[1].tostring()
    imageSize = len(img_str)
    val = struct.pack('!i', imageSize)
    socket.send(val)
    socket.send(img_str)
    print (imageSize)


import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import scipy.misc
import cv2
import numpy.ma as ma
#colors = np.random.randint(180, size=(80, 3))
colors = []
colors.append([180,255,255])
colors.append([150,255,255])
colors.append([120,255,255])
colors.append([90,255,255])
colors.append([60,255,255])
colors.append([30,255,255])
colors.append([0,255,255])
colors_for_boxes = np.random.randint(180, size=(100, 3))

def draw_human_body_parts(step, image, name='', image_height=1, image_width=1, bbox=None, label=None, gt_label=None, prob=None,final_mask=None):
    import cv2
    #image = image[0,...]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_body = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if bbox is not None:
        dictinary = {}
        for i, box in enumerate(bbox):
            width = int(box[2])-int(box[0])
            height = int(box[3])-int(box[1])
            #l=label[i]
            #p = prob[i,label[i]]
            if (prob[i,label[i]] > 0.98) and width*height >1000 and label[i]!=0:
                area = float((box[2]-box[0])*(box[3]-box[1]))
                while area in dictinary:
                    area+=1

                mask = final_mask[i]
                masks = np.zeros((height,width,7))
                body_mask = mask[...,0] > 0.6
                body_mask2 = np.array(body_mask,np.uint8)
                masks[...,0] = scipy.misc.imresize(body_mask2,(height,width))
                # cv2.imshow("body_mask",body_mask.astype(np.uint8)*255)
                # cv2.waitKey(3000)
                random_color_gauss =np.random.randint(0,180)
                for x in range(1,7):
                    maska = mask[...,x] > 0.6
                    # maska = np.logical_and(maska,body_mask)
                    # maska = ma.masked_array(mask[...,x], mask=np.logical_not(maska))
                    # maska = np.ma.filled(maska, 0)
                    #maska = maska >0
                    maska = scipy.misc.imresize(maska,(height,width))
                    masks[...,x] = maska
                dictinary[round(area,4)]=(box,label[i],1,prob[i,label[i]],masks,colors_for_boxes[i])
        sorted_keys = sorted(dictinary.iterkeys(),reverse=True)

        for key,i in zip(sorted_keys,range(len(sorted_keys))):
            bo, lab,gt_lab,_,mask,col= dictinary[key]

            max_indices = np.argmax(mask,axis=2)

            #random_color_gauss =np.random.randint(0,180) #for different boxes
            for x in range(int(bo[0]),int(bo[2])):
                for y in range(int(bo[1]),int(bo[3])):

                    xm = x-(int(bo[0]))
                    ym = y-(int(bo[1]))
                    if mask[ym,xm,0] >0:
                        hsv_body[y,x,0] = 120
                        hsv_body[y,x,1] = 190
                    if mask[ym,xm,max_indices[ym,xm]] >0:
                        hsv[y,x,0] = colors[max_indices[ym,xm]][0]
                        hsv[y,x,1] = 255

            contours,_ = cv2.findContours(mask[...,0].copy().astype(np.uint8), 1, 2)
            bigContour = None
            area = 0
            for c in contours:
                area2 = cv2.contourArea(c)
                if area2 > area:
                    area = area2
                    bigContour = c
            for p in bigContour:
                p[0,0] = p[0,0] + int(bo[0])
                p[0,1] = p[0,1] + int(bo[1])
            cv2.drawContours(hsv_body, [bigContour], 0, (0,255,0), 3)

        hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        hsv_body = cv2.cvtColor(hsv_body, cv2.COLOR_HSV2RGB)

        i=0
        for key in sorted_keys:
            bo, lab,gt_lab,_,_,col= dictinary[key]
            c = (255,0,0)
            bo, lab,gt_lab,_,_,col= dictinary[key]
            text = cat_id_to_cls_name(lab)
            i=i+1

            #hsv = cv2.rectangle(hsv,(int(bo[0]),int(bo[1])),(int(bo[2]),int(bo[3])),c,3)
            cv2.rectangle(hsv,(int(bo[0]),int(bo[1])),(int(bo[2]),int(bo[3])),c,3)
            cv2.rectangle(hsv_body,(int(bo[0]),int(bo[1])),(int(bo[2]),int(bo[3])),c,3)
            #hsv = cv2.putText(hsv,text+' '+str(i),(2+int(bo[0]),2+int(bo[1])), cv2.FONT_HERSHEY_SIMPLEX,0.5, color =(255,255,255))
            cv2.putText(hsv,text+' '+str(i),(2+int(bo[0]),2+int(bo[1])), cv2.FONT_HERSHEY_SIMPLEX,0.5, color =(255,255,255))
            cv2.putText(hsv_body,text+' '+str(i),(2+int(bo[0]),2+int(bo[1])), cv2.FONT_HERSHEY_SIMPLEX,0.5, color =(255,255,255))

    return hsv_body

def train():

    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #serversocket.bind(('10.20.5.205', 5555))
    serversocket.bind(('10.12.5.208', 5555))

    """The main function that runs training"""

    # data

    # image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = \
    #     datasets.get_dataset(FLAGS.dataset_name,
    #                          FLAGS.dataset_split_name,
    #                          FLAGS.dataset_dir,
    #                          FLAGS.im_batch,
    #                          is_training=True)
    ####o chestie importanta e ca le face resize la a.i. o latura sa nu fie mai mica de 640 pixeli


    # data_queue = tf.RandomShuffleQueue(capacity=32, min_after_dequeue=16,
    #         dtypes=(
    #             image.dtype, ih.dtype, iw.dtype,
    #             gt_boxes.dtype, gt_masks.dtype,
    #             num_instances.dtype, img_id.dtype))
    #enqueue_op = data_queue.enqueue((image, ih, iw, gt_boxes, gt_masks, num_instances, img_id))
    #data_queue_runner = tf.train.QueueRunner(data_queue, [enqueue_op] * 4)
    #tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, data_queue_runner)
    #(image, ih, iw, gt_boxes, gt_masks, num_instances, img_id) =  data_queue.dequeue()
    #im_shape = tf.shape(image)
    #image = tf.reshape(image, (im_shape[0], im_shape[1], im_shape[2], 3))
    from libs.preprocessings import coco_v1
    imageo, ih, iw, gt_boxes, gt_masks, num_instances, img_id = get_placeholders()
    image = tf.cast(imageo, tf.float32)
    image = image / 255.0
    image = (image - 0.5) * 2.0
    image = tf.expand_dims(image, axis=0)
    image = tf.reverse(image, axis=[-1])
    #image,gt_boxes,gt_masks = coco_v1.preprocess_for_training(image, gt_boxes, gt_masks)
    im_shape = tf.shape(image)
    image = tf.reshape(image, (im_shape[0], im_shape[1], im_shape[2], 3))

    ## network
    logits, end_points, pyramid_map = network.get_network(FLAGS.network, image,
            weight_decay=FLAGS.weight_decay, is_training=True)
    outputs = pyramid_network.build(end_points, im_shape[1], im_shape[2], pyramid_map,
            num_classes=2,
            base_anchors=9,
            is_training=False,
            gt_boxes=None, gt_masks=None,
            #loss_weights=[0.2, 0.2, 1.0, 0.2, 1.0]
            loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0])



    
    input_image = end_points['input']
    final_box = outputs['final_boxes']['box']
    final_cls = outputs['final_boxes']['cls']
    final_prob = outputs['final_boxes']['prob']
    if FLAGS.INCLUDE_MASK:
        final_mask = outputs['mask']['final_mask_for_drawing']

    ## solvers
    global_step = slim.create_global_step()
    #update_op = solve(global_step)
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
            )
    sess.run(init_op)

    # summary_op = tf.summary.merge_all()
    # logdir = os.path.join(FLAGS.train_dir, strftime('%Y%m%d%H%M%S', gmtime()))
    # if not os.path.exists(logdir):
    #     os.makedirs(logdir)
    #summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

    ## restore
    restore(sess)

    ## main loop
    # coord = tf.train.Coordinator()
    # threads = []
    # for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
    #     threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
    #                                      start=True))
    #
    # tf.train.start_queue_runners(sess=sess, coord=coord)
    #saver = tf.train.Saver(max_to_keep=20)

    print ("FLAGS INCLUDE MASK IS ",FLAGS.INCLUDE_MASK)



    print ("going to listen to socket")
    serversocket.listen(5)
    (clientsocket, address) = serversocket.accept()
    print ("new client")
    options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
    example = tf.train.Example()
    #for record in tf.python_io.tf_record_iterator('data/coco/records/out_VOC.tfrecord',options):
    while True:
        imagenp, gt_boxesnp,ihnp,iwnp,img_idnp = getImageFromSocket(clientsocket)
        ##########################################
        # example.ParseFromString(record)
        # imagenp, ihnp, iwnp, gt_boxesnp, gt_masksnp, num_instancesnp, img_idnp = get_npy_arrays(example)
        ##########################################

        #print (type(imagenp),type(gt_boxesnp),type(ihnp),type(iwnp),type(img_idnp))
        #imagenp = imagenp[np.newaxis]
        start_time = time.time()
        lista = [img_id] + [input_image] + [final_box] + [final_cls] + [final_prob] + [final_mask]
        feed_dict = {imageo:imagenp, ih:ihnp, iw:iwnp, img_id:img_idnp }
        img_idnp, input_imagenp, final_boxnp, final_clsnp, final_probnp, final_masknp= sess.run(lista,feed_dict=feed_dict)

        #final_gt_clsnp = np.argmax(np.asarray(final_gt_clsnp),axis=1)
        input_imagenp = np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0)
        #print (input_imagenp.shape, type(input_imagenp),input_imagenp.dtype)
        imagenp = draw_human_body_parts(1,input_imagenp,name="seg",bbox=final_boxnp,label=final_clsnp,gt_label=None,prob=final_probnp,final_mask=final_masknp)


        #print (imagenp.shape)
        print ("SHAPE",final_boxnp.shape,final_clsnp)
        sendImageThroughSocket(clientsocket,imagenp)
        duration_time = time.time() - start_time
        print ( """ image-id:%07d, time:%.3f(sec) """
                """ proposals: %d """
               % ( img_idnp, duration_time, len(final_boxnp)))




if __name__ == '__main__':
    train()
