# -*- coding:utf-8 -*-

'''
author: "caihua"
date: 2020/4/22
Email: caihua@datagrand.com
'''

import numpy as np
import os
import time
import logging
import cv2

from_ch = [
    u'，', u'！', u'：', u'（', u'）', u'；', u'—', u'“', u'”', u'‘', u'’', u'～', u'①', u'②', u'③', u'④', u'⑤', u'⑥', u'⑦',
    u'⑧', u'⑨', u'⑩', u'√', u'℃'
]
to_ch = [
    u',', u'!', u':', u'(', u')', u';', u'-', u'"', u'"', u'\'', u'\'', u'~', u'1', u'2', u'3', u'4', u'5', u'6', u'7',
    u'8', u'9', u'10', u'V', u'C'
]


def replace_character(text):
    result_text = text
    for i, c in enumerate(from_ch):
        if c in result_text:
            result_text = result_text.replace(c, to_ch[i])

    return result_text

class CrnnData(object):
    def __init__(self, data_dir_list, collate_fn, shuffle=True):
        self.data_path = []
        for data_dir in data_dir_list:
            print(data_dir)
            self.data_path.append(data_dir)

        self.load_dataset(self.data_path)
        self.collate = collate_fn
        self.start = 0

        if shuffle:
            self.shuffle()



    def load_dataset(self, data_path):
        self._data_list = None
        for rpath in data_path:
            if not os.path.exists(rpath):
                raise OSError('path not exists: {}'.format(rpath))
            try:
                tbeg = time.time()
                data_list = self._parse_data(rpath, is_replace_ch=True)
                if self._data_list is None:
                    self._data_list = data_list
                else:
                    self._data_list = np.vstack((self._data_list,data_list))
                print('parsing {} cost {} seconds'.format(rpath, time.time() - tbeg))
            except Exception as e:
                print(e)
        assert isinstance(self._data_list, np.ndarray) , "data list not in ndarray format"
        self.data_size = self._data_list.shape[0]
        self.data_indexes_array = np.arange(self.data_size)
        print('dataset size:', self.data_size)

    def _parse_data(self,data_path, is_replace_ch):
        data_list = []
        label_file_path = os.path.join(data_path, 'label.txt')
        with open(label_file_path, 'r') as f:
            for line in f:
                p = line.strip('\n').split()
                img_path = os.path.join(data_path, p[0])
                label_text = ' '.join(p[1:])
                if is_replace_ch:
                    label_text = replace_character(label_text)
                data_list.append((img_path, label_text))
        return np.asarray(data_list)

    def next_batch(self, batch_size):
        end = self.start + batch_size
        if end >= self.data_size:
            self.start = 0
            end = self.start + batch_size
        start = self.start
        images_batch, labels_batch, paths_batch = self.collate(self._data_list[start:end])
        try:
            assert images_batch.shape[0] == labels_batch.shape[0] == batch_size
        except Exception as e:
            return self.next_batch(batch_size)
        self.start += batch_size

        return images_batch, labels_batch, paths_batch

    def shuffle(self):
        for i in range(3):
            np.random.shuffle(self.data_indexes_array)
        self._data_list = self._data_list[self.data_indexes_array]

    @property
    def size(self):
        return len(self.data_size)

    def the_label(self, indexs):
        labels = []
        for i in indexs:
            labels.append(self.labels[i])
        return labels

class SoftpaddingCollate(object):
    def __init__(self, imgH=48, imgW=800, keep_ratio=True,nc=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.nc = nc

    def __call__(self,data_batch):
        image_paths = data_batch[:,0]
        labels_raw = data_batch[:,1]
        images = []
        labels = []
        image_paths_new = []
        for idx, img_path in enumerate(image_paths):
            try:
                img = cv2.imread(img_path)
                images.append(img)
                labels.append(labels_raw[idx])
                image_paths_new.append(image_paths[idx])
            except Exception as e:
                logging.info('corrupt image {}'.format(img_path))

        transform = PaddingResizeNormalize((self.imgW, self.imgH), keep_ratio=self.keep_ratio, nc=self.nc)
        images = [transform(image) for image in images]
        return np.asarray(images), np.asarray(labels), np.asarray(image_paths_new)

class PaddingResizeNormalize(object):

    def __init__(self, size, keep_ratio=True, nc=1):
        self.size = size
        self.keep_ratio = keep_ratio
        self.nc = nc

    def __call__(self, img):
        img = self.padding_and_resize(img,self.size[0], self.size[1], keep_ratio=self.keep_ratio, nc=self.nc)
        mean, std = cv2.meanStdDev(img)
        img = (img - mean[0][0]) / std[0][0]
        norm_img = np.zeros(img.shape)
        norm_img = cv2.normalize(img, norm_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img = np.expand_dims(norm_img,axis=-1) if len(norm_img.shape) == 2 else norm_img
        return norm_img

    def padding_and_resize(self, img, target_w, target_h, keep_ratio=True, nc=1):
        if nc == 1:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        scale_h = target_h * 1.0 / h

        t_w = max(1, int(w * scale_h)) if keep_ratio else min(w, target_w)

        if t_w >= target_w:
            img = cv2.resize(img,(target_w, target_h), cv2.INTER_CUBIC)
        else:
            img = cv2.resize(img,(t_w, target_h), cv2.INTER_CUBIC)

        chanel = 1 if len(img.shape) == 2 else 3
        new_img = np.full((target_h, target_w, chanel),255, np.uint8)
        h_final, w_final = img.shape[:2]
        img = np.expand_dims(img,axis=-1) if chanel == 1 else img
        new_img[:h_final,:w_final]= img.copy()
        return new_img

def test():
    data_dir = ['../data/test_images/train_data','../data/test_images/train_data']
    spc = SoftpaddingCollate()
    dataset = CrnnData(data_dir,spc)
    batch_data = dataset.next_batch(10)
    print(batch_data)


if __name__ == '__main__':
    test()
