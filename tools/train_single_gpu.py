# -*- coding:utf-8 -*-

'''
author: "caihua"
date: 2020/4/21
Email: caihua@datagrand.com
'''
import time
import copy
import os
import sys
import os.path as ops

PRO_PATH = ops.dirname(ops.dirname(ops.abspath(__file__)))
sys.path.append(PRO_PATH)
print(PRO_PATH)

import tensorflow as tf
import glog as logger
import numpy as np
from data_provider.data_reader import SoftpaddingCollate, CrnnData
from config import global_config
from crnn_model import crnn_net
from local_utils.establish_char_dict import CharDictBuilder
from local_utils import evaluation_tools

CFG = global_config.cfg

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

collate_fn = SoftpaddingCollate(imgH=48, imgW=800, keep_ratio=True, nc=1)


def train_shadownet(train_dataset_dir, val_dataset_dir, charset_path, weights_path=None, need_decode=False):
    """

    :param dataset_dir:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :param need_decode:
    :return:
    """

    # set learning rate
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate=CFG.TRAIN.LEARNING_RATE,
        global_step=global_step,
        decay_steps=CFG.TRAIN.LR_DECAY_STEPS,
        decay_rate=CFG.TRAIN.LR_DECAY_RATE,
        staircase=CFG.TRAIN.LR_STAIRCASE)

    # declare crnn net
    shadownet = crnn_net.ShadowNet(config=CFG, phase='train', charset_path=charset_path)
    inference_ret, ctc_loss, decoded, log_probm, seq_dist, inputs, targets, seq_len = shadownet.compute_loss(
        name='ctc_loss')
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss=ctc_loss,
                                                                                               global_step=global_step)

    # accuracy
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(inference_ret, seq_len, merge_repeated=False)
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
    # Set tf summary
    merge_summary_op, sess, summary_writer = summary_save(learning_rate, need_decode, seq_dist, ctc_loss)

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    saver = tf.train.Saver()

    # provider dataset
    train_dataset = CrnnData(data_dir_list=train_dataset_dir, collate_fn=collate_fn, shuffle=True)
    val_dataset = CrnnData(data_dir_list=val_dataset_dir, collate_fn=collate_fn, shuffle=True)
    char_build = CharDictBuilder(charset_path=charset_path)

    def encode_to_int(labels):
        label_encode, label_length = [], []
        for label in labels:
            label_t, label_tl = char_build.encode_label(label)
            label_encode.append(label_t)
            label_length.append(label_tl)
        label_encode = char_build.sparse_label(label_encode)
        return label_encode, label_length

    def do_report():
        val_images, val_labels, val_images_paths = val_dataset.next_batch(batch_size=CFG.TEST.BATCH_SIZE)
        val_labels, val_labels_length = encode_to_int(val_labels)
        feed_dict_val = {shadownet.inputs_x: val_images, shadownet.inputs_y: val_labels,
                         shadownet.seq_len: CFG.ARCH.SEQ_LENGTH * np.ones(CFG.TEST.BATCH_SIZE)}
        val_ctc_loss_value, val_seq_dist_value, avg_val_accuracy = sess.run(
            [ctc_loss, seq_dist, acc], feed_dict=feed_dict_val)

        return val_ctc_loss_value, val_seq_dist_value, avg_val_accuracy

    def train_batch(epoch, decode_flag=False):
        # prepare batch dataset
        train_images, train_labels, train_images_paths = train_dataset.next_batch(batch_size=CFG.TRAIN.BATCH_SIZE)
        # encode label to int
        train_labels, train_labels_length = encode_to_int(train_labels)
        feed_dict_train = {shadownet.inputs_x: train_images, shadownet.inputs_y: train_labels,
                           shadownet.seq_len: CFG.ARCH.SEQ_LENGTH * np.ones(CFG.TRAIN.BATCH_SIZE)}

        if decode_flag:
            _, train_ctc_loss_value, train_seq_dist_value, merge_summary_value, avg_train_accuracy = sess.run(
                [optimizer, ctc_loss, seq_dist, merge_summary_op, acc], feed_dict=feed_dict_train)
            print('Epoch_Train:', epoch, end=' ')
            print('Cost:', train_ctc_loss_value, end=' ')
            print('Seq_Distance:', train_seq_dist_value, end=' ')
            print('Train_Accuarcy:', avg_train_accuracy)
            # logger.info('Epoch_Train: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
            #     epoch + 1, train_ctc_loss_value, train_seq_dist_value, avg_train_accuracy[0]))
            if epoch % CFG.TRAIN.VAL_DISPLAY_STEP == 0:
                # validation part
                val_ctc_loss_value, val_seq_dist_value, avg_val_accuracy = do_report()
                print('Epoch_Val:', epoch, end=' ')
                print('Cost_Val:', val_ctc_loss_value, end=' ')
                print('Seq_Distance:', val_seq_dist_value, end=' ')
                print('Val_Accuarcy:', avg_val_accuracy)
            # logger.info('Epoch_Val: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
            #     epoch + 1, val_ctc_loss_value, val_seq_dist_value, avg_val_accuracy))
        else:
            _, train_ctc_loss_value, merge_summary_value = sess.run([optimizer, ctc_loss, merge_summary_op],
                                                                    feed_dict=feed_dict_train)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                logger.info('Epoch_Train: {:d} cost= {:9f}'.format(epoch + 1, train_ctc_loss_value))
        return train_ctc_loss_value, merge_summary_value

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
        min_cost = -1
        cost_history = [np.inf]

        while epoch < train_epochs:
            epoch += 1
            # setup early stopping
            patience_counter, break_flag = early_stop(epoch, cost_history, patience_counter)
            if break_flag: break
            train_ctc_loss_value, merge_summary_value = train_batch(epoch=epoch, decode_flag=True)

            # record history train ctc loss
            cost_history.append(train_ctc_loss_value)
            # add training sumary
            summary_writer.add_summary(summary=merge_summary_value, global_step=epoch)
            if epoch % 1000 == 0 and min_cost > train_ctc_loss_value:
                min_cost = train_ctc_loss_value
                model_save_path = model_save(epoch, min_cost)
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

    return np.array(cost_history[1:])  # Don't return the first np.inf


def summary_save(learning_rate, need_decode, seq_dist, ctc_loss, is_train=True, tboard_save_dir='tboard/crnn_syn90k'):
    os.makedirs(tboard_save_dir, exist_ok=True)
    name_loss = 'train_ctc_loss' if is_train else 'val_ctc_loss'
    tf.summary.scalar(name=name_loss, tensor=ctc_loss)
    tf.summary.scalar(name='learning_rate', tensor=learning_rate)
    if need_decode:
        name_dist = 'train_seq_distance' if is_train else 'val_seq_distance'
        tf.summary.scalar(name=name_dist, tensor=seq_dist)
    merge_summary_op = tf.summary.merge_all()

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)
    summary_writer = tf.summary.FileWriter(tboard_save_dir)
    summary_writer.add_graph(sess.graph)
    return merge_summary_op, sess, summary_writer


def model_save(epoch, cost, model_save_dir='model/crnn_syn90k'):
    # Set saver configuration
    os.makedirs(model_save_dir, exist_ok=True)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{}_{}_{:s}.ckpt'.format(epoch, cost, str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)
    return model_save_path


def early_stop(epoch, cost_history, patience_counter):
    break_flag = False
    # We always compare to the first point where cost didn't improve
    if (epoch < 1) and (cost_history[-1 - patience_counter] - cost_history[-1] > CFG.TRAIN.PATIENCE_DELTA):
        patience_counter = 1
    else:
        patience_counter += 1
    if patience_counter > CFG.TRAIN.PATIENCE_EPOCHS:
        logger.info("Cost didn't improve beyond {:f} for {:d} epochs, stopping early.".format(CFG.TRAIN.PATIENCE_DELTA,
                                                                                              patience_counter))
        break_flag = True
    return patience_counter, break_flag


if __name__ == '__main__':
    train_dataset_dir = val_dataset_dir = [os.path.join(PRO_PATH, 'data/test_images/train_data')]
    charset_path = os.path.join(PRO_PATH, 'data/test_images/doc_charset.txt')
    train_shadownet(train_dataset_dir, val_dataset_dir, charset_path, weights_path=None, need_decode=True)
