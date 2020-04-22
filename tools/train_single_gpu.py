# -*- coding:utf-8 -*-

'''
author: "caihua"
date: 2020/4/21
Email: caihua@datagrand.com
'''
import time
import os
import os.path as ops
import tensorflow as tf
import glog as logger
import numpy as np
from data_provider.data_reader import SoftpaddingCollate, CrnnData
from config import global_config
from crnn_model import crnn_net
from local_utils.establish_char_dict import CharDictBuilder
from data_provider import tf_io_pipline_fast_tools
from local_utils import evaluation_tools
CFG = global_config.cfg

collate_fn = SoftpaddingCollate(imgH=48,imgW=800,keep_ratio=True,nc=1)

def train_shadownet(train_dataset_dir, val_dataset_dir, charset_path, weights_path=None,  need_decode=False):
    """

    :param dataset_dir:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :param need_decode:
    :return:
    """
    # prepare dataset
    train_dataset = CrnnData(data_dir_list=train_dataset_dir,collate_fn=collate_fn,shuffle=True)
    val_dataset = CrnnData(data_dir_list=val_dataset_dir,collate_fn=collate_fn,shuffle=True)
    train_images, train_labels, train_images_paths = train_dataset.next_batch(batch_size=CFG.TRAIN.BATCH_SIZE)
    val_images, val_labels, val_images_paths = val_dataset.next_batch(batch_size=CFG.TRAIN.BATCH_SIZE)

    # encode label to int
    char_build = CharDictBuilder(charset_path=charset_path)
    train_label_encode , train_label_length = [] , []
    val_label_encode, val_label_length = [], []
    assert train_labels.shape[0] == val_labels.shape[0]
    for train_label, val_label in zip(train_labels,val_labels):
        label_t, label_tl = char_build.encode_label(train_label)
        train_label_encode.append(label_t)
        train_label_length.append(label_tl)
        label_v, label_vl = char_build.encode_label(val_label)
        val_label_encode.append(label_v)
        val_label_length.append(label_vl)

    # declare crnn net
    shadownet = crnn_net.ShadowNet(config=CFG,phase='train',charset_path=charset_path)
    shadownet_val = crnn_net.ShadowNet(config=CFG, phase='eval',charset_path=charset_path)

    # set up training graph
    with tf.device('/gpu:1'):

        # compute loss and seq distance
        train_inference_ret, train_ctc_loss = shadownet.compute_loss(
            inputdata=train_images,
            labels=train_labels,
            name='shadow_net',
            reuse=False
        )
        val_inference_ret, val_ctc_loss = shadownet_val.compute_loss(
            inputdata=val_images,
            labels=val_labels,
            name='shadow_net',
            reuse=True
        )

        train_decoded, train_log_prob = tf.nn.ctc_beam_search_decoder(
            train_inference_ret,
            CFG.ARCH.SEQ_LENGTH * np.ones(CFG.TRAIN.BATCH_SIZE),
            merge_repeated=False
        )
        val_decoded, val_log_prob = tf.nn.ctc_beam_search_decoder(
            val_inference_ret,
            CFG.ARCH.SEQ_LENGTH * np.ones(CFG.TRAIN.BATCH_SIZE),
            merge_repeated=False
        )

        train_sequence_dist = tf.reduce_mean(
            tf.edit_distance(tf.cast(train_decoded[0], tf.int32), train_labels),
            name='train_edit_distance'
        )
        val_sequence_dist = tf.reduce_mean(
            tf.edit_distance(tf.cast(val_decoded[0], tf.int32), val_labels),
            name='val_edit_distance'
        )

        # set learning rate
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(
            learning_rate=CFG.TRAIN.LEARNING_RATE,
            global_step=global_step,
            decay_steps=CFG.TRAIN.LR_DECAY_STEPS,
            decay_rate=CFG.TRAIN.LR_DECAY_RATE,
            staircase=CFG.TRAIN.LR_STAIRCASE)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9).minimize(
                loss=train_ctc_loss, global_step=global_step)

    # Set tf summary
    tboard_save_dir = 'tboard/crnn_syn90k'
    os.makedirs(tboard_save_dir, exist_ok=True)
    tf.summary.scalar(name='train_ctc_loss', tensor=train_ctc_loss)
    tf.summary.scalar(name='val_ctc_loss', tensor=val_ctc_loss)
    tf.summary.scalar(name='learning_rate', tensor=learning_rate)

    if need_decode:
        tf.summary.scalar(name='train_seq_distance', tensor=train_sequence_dist)
        tf.summary.scalar(name='val_seq_distance', tensor=val_sequence_dist)

    merge_summary_op = tf.summary.merge_all()

    # Set saver configuration
    saver = tf.train.Saver()
    model_save_dir = 'model/crnn_syn90k'
    os.makedirs(model_save_dir, exist_ok=True)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tboard_save_dir)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    with sess.as_default():
        epoch = 0
        if weights_path is None:
            logger.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info('Restore model from {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)
            epoch = sess.run(tf.train.get_global_step())

        patience_counter = 1
        cost_history = [np.inf]
        while epoch < train_epochs:
            epoch += 1
            # setup early stopping
            if epoch > 1 and CFG.TRAIN.EARLY_STOPPING:
                # We always compare to the first point where cost didn't improve
                if cost_history[-1 - patience_counter] - cost_history[-1] > CFG.TRAIN.PATIENCE_DELTA:
                    patience_counter = 1
                else:
                    patience_counter += 1
                if patience_counter > CFG.TRAIN.PATIENCE_EPOCHS:
                    logger.info("Cost didn't improve beyond {:f} for {:d} epochs, stopping early.".
                                format(CFG.TRAIN.PATIENCE_DELTA, patience_counter))
                    break

            if need_decode and epoch % 500 == 0:
                # train part
                _, train_ctc_loss_value, train_seq_dist_value, \
                    train_predictions, train_labels_sparse, merge_summary_value = sess.run(
                     [optimizer, train_ctc_loss, train_sequence_dist,
                      train_decoded, train_labels, merge_summary_op])

                train_labels_str = char_build.decode_label(train_labels_sparse)
                train_predictions = char_build.decode_label(train_predictions[0])
                avg_train_accuracy = evaluation_tools.compute_accuracy(train_labels_str, train_predictions)

                if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                    logger.info('Epoch_Train: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                        epoch + 1, train_ctc_loss_value, train_seq_dist_value, avg_train_accuracy))

                # validation part
                val_ctc_loss_value, val_seq_dist_value, \
                    val_predictions, val_labels_sparse = sess.run(
                     [val_ctc_loss, val_sequence_dist, val_decoded, val_labels])

                val_labels_str = char_build.decode_label(val_labels_sparse)
                val_predictions = char_build.decode_label(val_predictions[0])
                avg_val_accuracy = evaluation_tools.compute_accuracy(val_labels_str, val_predictions)

                if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                    logger.info('Epoch_Val: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                        epoch + 1, val_ctc_loss_value, val_seq_dist_value, avg_val_accuracy))
            else:
                _, train_ctc_loss_value, merge_summary_value = sess.run(
                    [optimizer, train_ctc_loss, merge_summary_op])

                if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                    logger.info('Epoch_Train: {:d} cost= {:9f}'.format(epoch + 1, train_ctc_loss_value))

            # record history train ctc loss
            cost_history.append(train_ctc_loss_value)
            # add training sumary
            summary_writer.add_summary(summary=merge_summary_value, global_step=epoch)

            if epoch % 1000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

    return np.array(cost_history[1:])  # Don't return the first np.inf

if __name__ == '__main__':
    train_dataset_dir = val_dataset_dir = '/Volumes/work/practice/CRNN_Tensorflow/data/test_images/train_data'
    charset_path = '/Volumes/work/practice/CRNN_Tensorflow/data/test_images/doc_charset.txt'
    train_shadownet(train_dataset_dir, val_dataset_dir, charset_path, weights_path=None, need_decode=False)