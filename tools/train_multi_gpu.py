# -*- coding:utf-8 -*-

'''
author: "caihua"
date: 2020/4/21
Email: caihua@datagrand.com
'''
import model
import time
import tensorflow as tf
import numpy as np
import os
import sys
import os.path as ops

PRO_PATH = ops.dirname(ops.dirname(ops.abspath(__file__)))
sys.path.append(PRO_PATH)
print(PRO_PATH)

import logging, datetime
from config import global_config
from data_provider.data_reader import CrnnData, SoftpaddingCollate
from local_utils.establish_char_dict import CharDictBuilder

FLAGS = global_config.cfg
logger = logging.getLogger('Traing for ocr using DenseCNN+BiLSTM+CTC')
logger.setLevel(logging.INFO)

collate_fn = SoftpaddingCollate(imgH=48, imgW=800, keep_ratio=True, nc=1)


def train(train_dir, val_dir, charset_path):
    g = model.Graph(is_training=True)
    print('loading train data, please wait---------------------', 'end= ')
    train_feeder = CrnnData(data_dir_list=train_dir, collate_fn=collate_fn, shuffle=True)
    val_feeder = CrnnData(data_dir_list=val_dir, collate_fn=collate_fn, shuffle=False)

    num_train_samples = train_feeder.size
    num_batches_per_epoch = int(num_train_samples / FLAGS.batch_size)
    num_val_samples = val_feeder.size
    num_val_per_epoch = int(num_val_samples / FLAGS.batch_size)

    char_build = CharDictBuilder(charset_path=charset_path)

    def encode_to_int(labels):
        label_encode, label_length = [], []
        for label in labels:
            label_t, label_tl = char_build.encode_label(label)
            label_encode.append(label_t)
            label_length.append(label_tl)
        label_encode = char_build.sparse_label(label_encode)
        return label_encode, label_length

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)
    with tf.Session(graph=g.graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        g.graph.finalize()
        train_writer = tf.summary.FileWriter('tboard/crnn_syn90k/train', sess.graph)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print('restore from the checkpoint{0}'.format(ckpt))

        print('=============================begin training=============================')
        val_inputs, val_labels, _ = val_feeder.next_batch(batch_size=FLAGS.TEST.BATCH_SIZE)
        val_labels, val_labels_length = encode_to_int(val_labels)
        # print(len(val_inputs))
        val_feed = {g.inputs: val_inputs,
                    g.labels: val_labels,
                    g.seq_len: np.array([g.cnn_time] * val_inputs.shape[0])}
        for cur_epoch in range(FLAGS.num_epochs):
            train_cost = 0
            batch_time = time.time()
            # the tracing part
            for cur_batch in range(num_batches_per_epoch):
                if (cur_batch + 1) % 100 == 0:
                    print('batch', cur_batch, ': time', time.time() - batch_time)
                batch_inputs, batch_labels, _ = train_feeder.next_batch(batch_size=FLAGS.TEST.BATCH_SIZE)
                batch_labels, train_labels_length = encode_to_int(batch_labels)
                feed = {g.inputs: batch_inputs,
                        g.labels: batch_labels,
                        g.seq_len: np.array([g.cnn_time] * batch_inputs.shape[0])}

                # if summary is needed
                # batch_cost,step,train_summary,_ = sess.run([cost,global_step,merged_summay,optimizer],feed)
                summary_str, batch_cost, step, _ = sess.run([g.merged_summay, g.cost, g.global_step, g.optimizer], feed)
                # calculate the cost
                train_cost += batch_cost * FLAGS.batch_size
                train_writer.add_summary(summary_str, step)

                # save the checkpoint
                if step % FLAGS.save_steps == 1000:
                    if not os.path.isdir(FLAGS.checkpoint_dir):
                        os.mkdir(FLAGS.checkpoint_dir)
                    logger.info('save the checkpoint of{0}', format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'), global_step=step)
                # train_err+=the_err*FLAGS.batch_size
                # do validation
                if step % FLAGS.validation_steps == 0:
                    dense_decoded, lastbatch_err, lr = sess.run([g.dense_decoded, g.lerr, g.learning_rate], val_feed)
                    # print the decode result
                    acc = accuracy_calculation(val_feeder.labels, dense_decoded, ignore_value=-1, isPrint=True)
                    avg_train_cost = train_cost / ((cur_batch + 1) * FLAGS.batch_size)
                    # train_err/=num_train_samples
                    now = datetime.datetime.now()
                    log = "{}/{} {}:{}:{} Epoch {}/{}, accuracy = {:.3f},avg_train_cost = {:.3f}, lastbatch_err = {:.3f}, lr={:.8f}"
                    print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                     cur_epoch + 1, FLAGS.num_epochs, acc, avg_train_cost, lastbatch_err, lr))


def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint=True):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq,please check again')
        return 0
    count = 0
    for i, origin_label in enumerate(original_seq):
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        if isPrint and i < 18:
            print('seq{0:4d}: origin: {1} decoded:{2}'.format(i, origin_label, decoded_label))
        if origin_label == decoded_label: count += 1
    return count * 1.0 / len(original_seq)


if __name__ == '__main__':
    train_dataset_dir = val_dataset_dir = [os.path.join(PRO_PATH, 'data/test_images/train_data')]
    charset_path = os.path.join(PRO_PATH, 'data/test_images/doc_charset.txt')
    train(train_dir=train_dataset_dir, val_dir=val_dataset_dir, charset_path=charset_path)
