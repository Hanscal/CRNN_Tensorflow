#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午1:39
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : train_shadownet.py
# @IDE: PyCharm Community Edition
"""
Train shadow net script
"""
import os
import os.path as ops
import time
import math
import argparse
import sys
sys.path.append('./')

import tensorflow as tf
import glog as logger
import numpy as np

from crnn_model import crnn_net
from local_utils import evaluation_tools
from config import global_config
from data_provider import shadownet_data_feed_pipline
from data_provider import tf_io_pipline_fast_tools

CFG = global_config.cfg


def init_args():
    """
    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_dir', type=str,
                        help='Directory containing train_features.tfrecords')
    parser.add_argument('-w', '--weights_path', type=str,
                        help='Path to pre-trained weights to continue training')
    parser.add_argument('-c', '--char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--ord_map_dict_path', type=str,
                        help='Directory where ord map dictionaries for the dataset were stored')
    parser.add_argument('-e', '--decode_outputs', type=args_str2bool, default=False,
                        help='Activate decoding of predictions during training (slow!)')
    parser.add_argument('-m', '--multi_gpus', type=args_str2bool, default=False,
                        nargs='?', const=True, help='Use multi gpus to train')

    return parser.parse_args()


if __name__ == '__main__':

    # init args
    args = init_args()

    if args.multi_gpus:
        logger.info('Use multi gpus to train the model')
        train_shadownet_multi_gpu(
            dataset_dir=args.dataset_dir,
            weights_path=args.weights_path,
            char_dict_path=args.char_dict_path,
            ord_map_dict_path=args.ord_map_dict_path
        )
    else:
        logger.info('Use single gpu to train the model')
        train_shadownet(
            dataset_dir=args.dataset_dir,
            weights_path=args.weights_path,
            char_dict_path=args.char_dict_path,
            ord_map_dict_path=args.ord_map_dict_path,
            need_decode=args.decode_outputs
        )
