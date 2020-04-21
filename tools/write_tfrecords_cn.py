# -*- coding:utf-8 -*-

'''
author: "caihua"
date: 2019/5/11
Email: hanscalcai@163.com
'''

"""
Write tfrecords tools
"""
import argparse
import os
import tensorflow as tf
import cv2
import time
import tqdm
import json
import glog as log
import sys
sys.path.append('../')
import os.path as ops

from config import global_config
CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, help='The dataset_dir')
    parser.add_argument('-l', '--label_dir', type=str, help='The label dir')
    parser.add_argument('-s', '--save_dir', type=str, help='The generated tfrecords save dir')
    parser.add_argument('-c', '--char_dict_path', type=str, default=None,
                        help='The char dict file path. If it is None the char dict will be'
                             'generated automatically in folder data/char_dict')
    parser.add_argument('-o', '--ord_map_dict_path', type=str, default=None,
                        help='The ord map dict file path. If it is None the ord map dict will be'
                             'generated automatically in folder data/char_dict')

    return parser.parse_args()

def _is_valid_jpg_file(image_path):
    """
    :param image_path:
    :return:
    """

    if not ops.exists(image_path):
        return False

    file = open(image_path, 'rb')
    data = file.read(11)
    if data[:4] != '\xff\xd8\xff\xe0' and data[:4] != '\xff\xd8\xff\xe1':
        file.close()
        return False
    if data[6:] != 'JFIF\0' and data[6:] != 'Exif\0':
        file.close()
        return False
    file.close()

    file = open(image_path, 'rb')
    file.seek(-2, 2)
    if file.read() != '\xff\xd9':
        file.close()
        return False

    file.close()

    return True

class FeatureIO(object):
    def __init__(self,char_dict_path,ord_map_dict_path):
        self._char_dict = self.read_char_dict(char_dict_path)
        self._ord_map = self.read_ord_map_dict(ord_map_dict_path)

    def _int64_feature(self, value):
        """
        Wrapper for inserting int64 features into Example proto.
        :param value:
        :return:
        """
        if not isinstance(value, list):
            value = [value]
        value_tmp = []
        is_int = True
        for val in value:
            if not isinstance(val, int):
                is_int = False
                value_tmp.append(int(float(val)))
        if not is_int:
            value = value_tmp
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _bytes_feature(self,value):
        """
        Wrapper for inserting bytes features into Example proto.
        :param value:
        :return:
        """
        if not isinstance(value, bytes):
            if not isinstance(value, list):
                value = value.encode('utf-8')
            else:
                value = [val.encode('utf-8') for val in value]
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def char_to_int(self, char):
        """
        convert char into int index, first convert the char into it's ord
        number and the convert the ord number into int index which is stored
        in ord_map_dict.json file
        :param char: single character
        :return: the int index of the character
        """
        str_key = str(ord(char)) + '_ord'
        try:
            result = int(self._ord_map[str_key])
            return result
        except KeyError:
            raise KeyError("Character {} missing in ord_map.json".format(char))

    def encode_label(self,label):
        encode_label = [self.char_to_int(char) for char in label]
        length = len(label)
        return encode_label, length

    @staticmethod
    def read_ord_map_dict(ord_map_dict_path):
        with open(ord_map_dict_path, 'r', encoding='utf-8') as json_f:
            res = json.load(json_f)
        return res

    @staticmethod
    def read_char_dict(char_dict_path):
        with open(char_dict_path, 'r', encoding='utf-8') as json_f:
            res = json.load(json_f)
        return res



    def write_tfrecords(self,dataset_dir,annotation_file_path,save_path):
        # establish train example info
        train_sample_infos = []
        log.info('Start initialize {} sample information list...'.format(save_path.split('/')[-1]))
        num_lines = sum(1 for _ in open(annotation_file_path, 'r'))
        with open(annotation_file_path, 'r', encoding='utf-8') as file:
            for line in tqdm.tqdm(file, total=num_lines):

                image_name, label = line.rstrip('\r').rstrip('\n').split()
                image_path = ops.join(dataset_dir, image_name)
                # import pdb;pdb.set_trace()
                if not ops.exists(image_path):
                    print ('Example image {:s} not exist'.format(image_path))
                    continue
                train_sample_infos.append((image_path, label))

        tfrecords_writer = tf.python_io.TFRecordWriter(path=save_path)
        for num,train_info in enumerate(train_sample_infos):
            sample_path, sample_label = train_info[0], train_info[1]
            if _is_valid_jpg_file(sample_path):
                log.error('Image file: {:d} is not a valid jpg file'.format(sample_path))
                continue

            try:
                image = cv2.imread(sample_path, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                image = cv2.resize(image, dsize=tuple(CFG.ARCH.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
                image = image.tostring()
            except IOError as err:
                log.error(err)
                continue

            (sample_label,length) = self.encode_label(sample_label)

            features = tf.train.Features(feature={
                'labels': self._int64_feature(sample_label),
                'images': self._bytes_feature(image),
                'imagepaths': self._bytes_feature(sample_path)
            })
            tf_example = tf.train.Example(features=features)
            tfrecords_writer.write(tf_example.SerializeToString())
            num+=1
            if num%1000==0:
                sys.stdout.write('\r>>Writing {:d}/{:d} at time: {}'.format(num + 1, len(train_sample_infos), time.strftime('%H:%M:%S')))
                sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        return

def run(dataset_dir,label_dir, save_dir):
    """
    Write tensorflow records for training , testing and validation
    :param dataset_dir: the root dir of crnn dataset
    :param char_dict_path: json file path which contains the map relation
    between ord value and single character
    :param ord_map_dict_path: json file path which contains the map relation
    between int index value and char ord value
    :param save_dir: the root dir of tensorflow records to write into
    :return:
    """
    assert ops.exists(dataset_dir), '{:s} not exist'.format(dataset_dir)

    os.makedirs(save_dir, exist_ok=True)
    _train_annotation_file_path = ops.join(label_dir, 'annotation_train.txt')
    _test_annotation_file_path = ops.join(label_dir, 'annotation_test.txt')
    _val_annotation_file_path = ops.join(label_dir, 'annotation_val.txt')
    _char_dict_path = ops.join('../data/char_dict', 'char_dict_cn.json')
    _ord_map_dict_path = ops.join('../data/char_dict', 'ord_map_cn.json')

    feature_io = FeatureIO(_char_dict_path,_ord_map_dict_path)
    feature_io.write_tfrecords(dataset_dir, _train_annotation_file_path, os.path.join(save_dir, 'train.tfrecords'))
    feature_io.write_tfrecords(dataset_dir, _test_annotation_file_path, os.path.join(save_dir, 'test.tfrecords'))
    feature_io.write_tfrecords(dataset_dir, _val_annotation_file_path, os.path.join(save_dir, 'val.tfrecords'))

if __name__ == '__main__':
    """
    generate tfrecords
    """
    args = init_args()

    run(
        dataset_dir=args.dataset_dir,
        label_dir=args.label_dir,
        save_dir=args.save_dir
    )