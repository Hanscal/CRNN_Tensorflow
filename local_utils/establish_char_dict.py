#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-30 下午4:01
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : establish_char_dict.py
# @IDE: PyCharm Community Edition
"""
Establish the char dictionary in order to contain chinese character
"""
import json
import os
import os.path as ops
import numpy as np
from typing import Iterable


class CharDictBuilder(object):
    """
        Build and read char dict
    """
    def __init__(self, charset_path=None):
        if charset_path is not None:
            self.char2id, self.id2char = self.read_charset(charset_path)

    @staticmethod
    def _read_chars(origin_char_list):
        """
        Read a list of chars or a file containing it.
        :param origin_char_list:
        :return:
        """
        if isinstance(origin_char_list, str):
            assert ops.exists(origin_char_list), \
                "Character list %s is not a file or could not be found" % origin_char_list
            with open(origin_char_list, 'r', encoding='utf-8') as origin_f:
                chars = (l[0] for l in origin_f.readlines())
        elif isinstance(origin_char_list, Iterable):
            ok = all(map(lambda s: isinstance(s, str) and len(s) == 1, origin_char_list))
            assert ok, "Character list is not an Iterable of strings of length 1"
            chars = origin_char_list
        else:
            raise TypeError("Character list needs to be a file or a list of strings")
        return chars

    @staticmethod
    def _write_json(save_path, data):
        """

        :param save_path:
        :param data:
        :return:
        """
        if not save_path.endswith('.json'):
            raise ValueError('save path {:s} should be a json file'.format(save_path))
        os.makedirs(ops.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as json_f:
            json.dump(data, json_f, sort_keys=True, indent=4)

    @staticmethod
    def write_char_dict(origin_char_list, save_path):
        """
        Writes the ordinal to char map used in int_to_char to decode predictions and labels.
        The file is read with CharDictBuilder.read_char_dict()
        :param origin_char_list: Either a path to file with character list, one a character per line, or a list or set
                                 of characters
        :param save_path: Destination file, full path.
        """
        char_dict = {str(ord(c)) + '_ord': c for c in CharDictBuilder._read_chars(origin_char_list)}
        CharDictBuilder._write_json(save_path, char_dict)

    @staticmethod
    def read_char_dict(dict_path):
        """

        :param dict_path:
        :return: a dict with ord(char) as key and char as value
        """
        with open(dict_path, 'r', encoding='utf-8') as json_f:
            res = json.load(json_f)
        return res

    @staticmethod
    def map_ord_to_index(origin_char_list, save_path):
        """
        Map ord of character in origin char list into index start from 0 in order to meet the output of the DNN
        :param origin_char_list:
        :param save_path:
        """
        ord_2_index_dict = {str(i) + '_index': str(ord(c)) for i, c in
                            enumerate(CharDictBuilder._read_chars(origin_char_list))}
        index_2_ord_dict = {str(ord(c)) + '_ord': str(i) for i, c in
                            enumerate(CharDictBuilder._read_chars(origin_char_list))}
        total_ord_map_index_dict = dict(ord_2_index_dict)
        total_ord_map_index_dict.update(index_2_ord_dict)
        CharDictBuilder._write_json(save_path, total_ord_map_index_dict)

    @staticmethod
    def read_ord_map_dict(ord_map_dict_path):
        """

        :param ord_map_dict_path:
        :return:
        """
        with open(ord_map_dict_path, 'r', encoding='utf-8') as json_f:
            res = json.load(json_f)
        return res

    @staticmethod
    def read_charset(charset_path):
        with open(charset_path,'r') as fr:
            data = fr.readline().strip('\n')
            char_list = list(data)
            char2id = {char: str(idx) for idx,char in enumerate(char_list)}
            id2char = {str(idx): char for idx,char in enumerate(char_list)}
        return char2id, id2char

    def sparse_label(self,texts):
        """Create a sparse representention of x.
           Args:
               sequences: a list of lists of type dtype where each element is a sequence
           Returns:
               A tuple with (indices, values, shape)
           """
        indices = []
        values = []

        for n, seq in enumerate(texts):
            indices.extend(zip([n] * len(seq), range(0, len(seq), 1)))
            values.extend(seq)
        indices = np.asarray(indices,dtype=np.int64)
        values = np.asarray(values, dtype=np.int32)
        shape = np.asarray([len(texts), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

        return indices, values, shape

    def char_to_int(self, char):
        try:
            result = int(self.char2id[char])
            return result
        except KeyError:
            raise KeyError("Character {} missing in ord_map.json".format(char))

    def int_to_char(self, num):
        try:
            result = self.id2char(num)
            return result
        except KeyError:
            raise KeyError("Character {} missing in ord_map.json".format(char))

    def encode_label(self,label):
        encode_label = [self.char_to_int(char) for char in label]
        length = len(label)
        return encode_label, length

    def decode_label(self, sparse_tensor):
        """
        :param sparse_tensor: prediction or ground truth label
        :return: String value of the sparse tensor
        """
        indices = sparse_tensor.indices
        values = sparse_tensor.values
        # Translate from consecutive numbering into char
        values = np.array([[self.int_to_char(num) for num in tmp ] for tmp in values])
        dense_shape = sparse_tensor.dense_shape

        number_lists = np.ones(dense_shape, dtype=values.dtype)
        str_lists = []
        res = []
        for i, index in enumerate(indices):
            number_lists[index[0], index[1]] = values[i]
        for number_list in number_lists:
            # append characters
            str_lists.append([val for val in number_list])
        for str_list in str_lists:
            # int_to_char() returns '\x00' for an input == 1, which is the default
            # value in number_lists, so we skip it when building the result
            res.append(''.join(c for c in str_list if c != '\x00'))
        return res