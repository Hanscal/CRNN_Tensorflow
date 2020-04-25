# -*- coding:utf-8 -*-

'''
author: "caihua"
date: 2020/4/25
Email: caihua@datagrand.com
'''

import tensorflow as tf
from config import global_config

CFG = global_config.cfg
import utils
import os, sys

slim = tf.contrib.slim
from TPS import ThinPlateSpline2 as stn

FLAGS = global_config.cfg
from crnn_model.densenet import *
from local_utils.establish_char_dict import CharDictBuilder


def stacked_bidirectional_rnn(RNN, num_units, num_layers, inputs, seq_lengths):
    """
    multi layer bidirectional rnn
    :param RNN: RNN class, e.g. LSTMCell
    :param num_units: int, hidden unit of RNN cell
    :param num_layers: int, the number of layers
    :param inputs: Tensor, the input sequence, shape: [batch_size, max_time_step, num_feature]
    :param seq_lengths: list or 1-D Tensor, sequence length, a list of sequence lengths, the length of the list is batch_size
    :param batch_size: int
    :return: the output of last layer bidirectional rnn with concatenating
    """
    # TODO: add time_major parameter, and using batch_size = tf.shape(inputs)[0], and more assert
    _inputs = inputs
    if len(_inputs.get_shape().as_list()) != 3:
        raise ValueError("the inputs must be 3-dimentional Tensor")

    for _ in range(num_layers):
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            rnn_cell_fw = RNN(num_units)
            rnn_cell_bw = RNN(num_units)
            (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs, seq_lengths,
                                                              dtype=tf.float32)
            _inputs = tf.concat(output, 2)

    return _inputs


class Graph(object):
    def __init__(self, charset_path,is_training=True,cfg=FLAGS):
        self.config = cfg
        self.imgW, self.imgH = self.config.INPUT_SIZE
        self.nc = self.config.INPUT_CHANNELS
        self.chardict = CharDictBuilder()
        self.char2id, self.id2char = self.chardict.read_charset(charset_path)
        self._num_classes = len(self.char2id)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [None, self.imgW, self.imgH, self.nc])
            '''with tf.variable_scope('STN'):
                #Localisation net
                conv1_loc = slim.conv2d(self.inputs, 32, [3, 3], scope='conv1_loc')
                pool1_loc = slim.max_pool2d(conv1_loc, [2, 2], scope='pool1_loc')
                conv2_loc = slim.conv2d(pool1_loc, 64, [3, 3], scope='conv2_loc')
                pool2_loc = slim.max_pool2d(conv2_loc, [2, 2], scope='pool2_loc')
                pool2_loc_flat = slim.flatten(pool2_loc)
                fc1_loc = slim.fully_connected(pool2_loc_flat, 1024, scope='fc1_loc')
                fc2_loc = slim.fully_connected(fc1_loc, 128, scope='fc2_loc')
                W = tf.Variable(tf.zeros([128, 20]))
                b = tf.Variable(initial_value=[-1, -0.2, -0.5, -0.35, 0, -0.5, 0.5, -0.67, 1, -0.8,
                                               -1, 0.8, -0.5, 0.65, 0, 0.5, 0.5, 0.33, 1, 0.2], dtype=tf.float32)
                # fc3_loc=tf.layers.dense(fc2_loc,20,activation=tf.nn.tanh,kernel_initializer=tf.zeros_initializer)
                # fc3_loc = slim.fully_connected(fc2_loc, 8, activation_fn=tf.nn.tanh, scope='fc3_loc')
                # spatial transformer
                fc3_loc = tf.nn.tanh(tf.matmul(fc2_loc, W) + b)
                loc = tf.reshape(fc3_loc, [-1, 10, 2])
                # spatial transformer
                s = np.array([[-0.95, -0.95], [-0.5, -0.95], [0, -0.95], [0.5, -0.95], [0.95, -0.95], [-0.95, 0.95], [-0.5, 0.95], [0, 0.95], [0.5, 0.95],
                              [0.95,0.95]] * 256)
                s = tf.constant(s.reshape([256, 10, 2]), dtype=tf.float32)
                self.h_trans = stn(self.inputs, s, loc, (utils.image_width, utils.image_height))'''
            with tf.variable_scope('Dense_CNN'):
                    nb_filter = 64
                    net = tf.layers.conv2d(self.inputs, nb_filter, 5, (2, 2), "SAME", use_bias=False)
                    net, nb_filter = dense_block(net, 8, 8, nb_filter, is_training)
                    net, nb_filter = transition_block(net, 128, is_training, pooltype=2)
                    net, nb_filter = dense_block(net, 8, 8, nb_filter, is_training)
                    net, nb_filter = transition_block(net, 128, is_training, pooltype=3)
                    net, nb_filter = dense_block(net, 8, 8, nb_filter, is_training)
                    # net, nb_filter = transition_block(net, 128, is_training, pooltype=3)
                    print(net)
                    # net = tf.layers.conv2d(net, nb_filter, 3, (1, 2), "SAME", use_bias=True)
                    self.cnn_time = net.get_shape().as_list()[1]
                    self.num_feauture = 4 * 192

            temp_inputs = net
            with tf.variable_scope('BiLSTM'):
                self.labels = tf.sparse_placeholder(tf.int32)
                self.seq_len = tf.placeholder(tf.int32, [None])
                self.lstm_inputs = tf.reshape(temp_inputs, [-1, self.cnn_time, self.num_feauture])
                outputs = stacked_bidirectional_rnn(tf.contrib.rnn.LSTMCell, FLAGS.num_hidden, 2, self.lstm_inputs,
                                                    self.seq_len)
            # The second output is the last state and we will no use that
            # outputs, _ = tf.nn.dynamic_rnn(stack, self.lstm_inputs, self.seq_len, dtype=tf.float32)
            shape = tf.shape(self.lstm_inputs)
            batch_s, max_timesteps = shape[0], shape[1]
            # Reshaping to apply the same weights over the timesteps
            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden * 2])
            W = tf.Variable(tf.truncated_normal([FLAGS.num_hidden * 2, self._num_classes], stddev=0.1, dtype=tf.float32),
                            name='W')
            b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[self._num_classes], name='b'))
            logits = tf.matmul(outputs, W) + b
            # Reshaping back to the original shape
            logits = tf.reshape(logits, [batch_s, -1, self._num_classes])
            # Time major
            logits = tf.transpose(logits, (1, 0, 2))
            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=self.seq_len)
            self.cost = tf.reduce_mean(self.loss)
            self.learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate, self.global_step,
                                                            FLAGS.decay_steps,
                                                            FLAGS.decay_rate, staircase=True)
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=FLAGS.momentum,
                                                        use_nesterov=True).minimize(self.cost,
                                                                                    global_step=self.global_step)

            # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
            # (it's slower but you'll get better results)
            # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len,merge_repeated=False)
            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, self.seq_len, merge_repeated=False)
            self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
            # Inaccuracy: label error rate
            self.lerr = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))

            tf.summary.scalar('cost', self.cost)
            # tf.summary.scalar('lerr',self.lerr)
            self.merged_summay = tf.summary.merge_all()