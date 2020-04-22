# -*- coding:utf-8 -*-

'''
author: "caihua"
date: 2020/4/21
Email: caihua@datagrand.com
'''
import time
import os
import os.path as ops
import math
import tensorflow as tf
import glog as logger
import numpy as np
from config import global_config
from crnn_model import crnn_net
from data_provider import shadownet_data_feed_pipline
CFG = global_config.cfg

def train_shadownet_multi_gpu(dataset_dir, weights_path, char_dict_path, ord_map_dict_path):
    """

    :param dataset_dir:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :return:
    """
    # prepare dataset information
    train_dataset = shadownet_data_feed_pipline.CrnnDataFeeder(
        dataset_dir=dataset_dir,
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path,
        flags='train'
    )
    val_dataset = shadownet_data_feed_pipline.CrnnDataFeeder(
        dataset_dir=dataset_dir,
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path,
        flags='val'
    )
    train_images, train_labels, train_images_paths = train_dataset.inputs(
        batch_size=CFG.TRAIN.BATCH_SIZE
    )
    val_images, val_labels, val_images_paths = val_dataset.inputs(
        batch_size=CFG.TRAIN.BATCH_SIZE
    )

    # set crnn net
    shadownet = crnn_net.ShadowNet(
        phase='train',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )
    shadownet_val = crnn_net.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )

    # set average container
    tower_grads = []
    train_tower_loss = []
    val_tower_loss = []
    batchnorm_updates = None
    train_summary_op_updates = None

    # set lr
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate=CFG.TRAIN.LEARNING_RATE,
        global_step=global_step,
        decay_steps=CFG.TRAIN.LR_DECAY_STEPS,
        decay_rate=CFG.TRAIN.LR_DECAY_RATE,
        staircase=CFG.TRAIN.LR_STAIRCASE)

    # set up optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

    # set distributed train op
    with tf.variable_scope(tf.get_variable_scope()):
        is_network_initialized = False
        for i in range(CFG.TRAIN.GPU_NUM):
            with tf.device('/gpu:{:d}'.format(i)):
                with tf.name_scope('tower_{:d}'.format(i)) as _:
                    train_loss, grads = compute_net_gradients(
                        train_images, train_labels, shadownet, optimizer,
                        is_net_first_initialized=is_network_initialized)

                    is_network_initialized = True

                    # Only use the mean and var in the first gpu tower to update the parameter
                    if i == 0:
                        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        train_summary_op_updates = tf.get_collection(tf.GraphKeys.SUMMARIES)

                    tower_grads.append(grads)
                    train_tower_loss.append(train_loss)
                with tf.name_scope('validation_{:d}'.format(i)) as _:
                    val_loss, _ = compute_net_gradients(
                        val_images, val_labels, shadownet_val, optimizer,
                        is_net_first_initialized=is_network_initialized)
                    val_tower_loss.append(val_loss)

    grads = average_gradients(tower_grads)
    avg_train_loss = tf.reduce_mean(train_tower_loss)
    avg_val_loss = tf.reduce_mean(val_tower_loss)

    # Track the moving averages of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(
        CFG.TRAIN.MOVING_AVERAGE_DECAY, num_updates=global_step)
    variables_to_average = tf.trainable_variables() + tf.moving_average_variables()
    variables_averages_op = variable_averages.apply(variables_to_average)

    # Group all the op needed for training
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
    train_op = tf.group(apply_gradient_op, variables_averages_op,
                        batchnorm_updates_op)

    # set tensorflow summary
    tboard_save_path = 'tboard/crnn_syn90k_multi_gpu'
    os.makedirs(tboard_save_path, exist_ok=True)

    summary_writer = tf.summary.FileWriter(tboard_save_path)

    avg_train_loss_scalar = tf.summary.scalar(name='average_train_loss',
                                              tensor=avg_train_loss)
    avg_val_loss_scalar = tf.summary.scalar(name='average_val_loss',
                                            tensor=avg_val_loss)
    learning_rate_scalar = tf.summary.scalar(name='learning_rate_scalar',
                                             tensor=learning_rate)
    train_merge_summary_op = tf.summary.merge(
        [avg_train_loss_scalar, learning_rate_scalar] + train_summary_op_updates
    )
    val_merge_summary_op = tf.summary.merge([avg_val_loss_scalar])

    # set tensorflow saver
    saver = tf.train.Saver()
    model_save_dir = 'model/crnn_syn90k_multi_gpu'
    os.makedirs(model_save_dir, exist_ok=True)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # set sess config
    sess_config = tf.ConfigProto(device_count={'GPU': CFG.TRAIN.GPU_NUM}, allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    logger.info('Global configuration is as follows:')
    logger.info(CFG)

    sess = tf.Session(config=sess_config)

    summary_writer.add_graph(sess.graph)

    with sess.as_default():
        epoch = 0
        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/shadownet_model.pb'.format(model_save_dir))

        if weights_path is None:
            logger.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)
            epoch = sess.run(tf.train.get_global_step())

        train_cost_time_mean = []
        val_cost_time_mean = []

        while epoch < train_epochs:
            epoch += 1
            # training part
            t_start = time.time()

            _, train_loss_value, train_summary, lr = \
                sess.run(fetches=[train_op,
                                  avg_train_loss,
                                  train_merge_summary_op,
                                  learning_rate])

            if math.isnan(train_loss_value):
                raise ValueError('Train loss is nan')

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)

            summary_writer.add_summary(summary=train_summary,
                                       global_step=epoch)

            # validation part
            t_start_val = time.time()

            val_loss_value, val_summary = \
                sess.run(fetches=[avg_val_loss,
                                  val_merge_summary_op])

            summary_writer.add_summary(val_summary, global_step=epoch)

            cost_time_val = time.time() - t_start_val
            val_cost_time_mean.append(cost_time_val)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                logger.info('Epoch_Train: {:d} total_loss= {:6f} '
                            'lr= {:6f} mean_cost_time= {:5f}s '.
                            format(epoch + 1,
                                   train_loss_value,
                                   lr,
                                   np.mean(train_cost_time_mean)
                                   ))
                train_cost_time_mean.clear()

            if epoch % CFG.TRAIN.VAL_DISPLAY_STEP == 0:
                logger.info('Epoch_Val: {:d} total_loss= {:6f} '
                            ' mean_cost_time= {:5f}s '.
                            format(epoch + 1,
                                   val_loss_value,
                                   np.mean(val_cost_time_mean)))
                val_cost_time_mean.clear()

            if epoch % 5000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
    sess.close()

    return