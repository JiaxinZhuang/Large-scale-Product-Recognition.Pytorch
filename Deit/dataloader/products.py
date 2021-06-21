# encoding=utf-8
import pdb

import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import os
from PIL import Image
import torch.utils.data.dataset as dataset
from torchvision import transforms
import torchvision.transforms as transforms

import random
import numpy as np
import cv2

from dataloader.transforms.custom_transform import read_image
from tfrecord.torch.dataset import MultiTFRecordDataset
import torch
import struct
from .tfrecord_utils import yt_example_pb2
from .tfrecord_utils.getImageData import getImageData

class TFRecordDataset(dataset.Dataset):

    def __init__(self, transforms, prng=np.random, split='train',version=""):
        #self.names = names
        self.tfrecord_root = '/dev/shm/'
        self.transforms = transforms
        self.split = split
        self.version = version
        # print(self.version)
        self.getFileList()  #self.filelist, self.labellist =
        #self.pre_process_im = PreProcessIm(prng=prng, **pre_process_im_kwargs)
        print(self.split, ' filelist ', len(self.filelist))

    def getFileList(self):
        self.filelist, self.labellist = [], []
        self.label_offset = 0
        #for idx in self.names:
        if self.split == 'train':
            index_filepath = self.tfrecord_root + 'TFR-aliproduct_' + self.split + self.version +  '.txt'
        elif self.split == 'val':
            index_filepath = self.tfrecord_root + 'TFR-aliproduct_' + self.split + '.txt'
        else:
            print('=> split', self.split)
            index_filepath = self.tfrecord_root + 'TFR-aliproduct_train' + self.version +  'val.txt'


        with open(index_filepath, "r") as idx_r:
            for line in idx_r:
                data_name, tf_num, offset, label = line.rstrip().split('\t')[:4]
                # file_name = '{0}*{1:05}*{2}'.format(data_name, int(tf_num), offset)
                file_name = (data_name, str(tf_num).zfill(5), offset)
                self.filelist.append(file_name)
                label = int(label) + self.label_offset
                self.labellist.append(label)
            self.label_offset = max(self.labellist)


    def get_file_loc(self, index):
        if len(self.filelist) <= index:
            print(index, len(self.filelist))
        return self.filelist[index]

    def get_label(self, index):
        return self.labellist[index]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        file_loc, label = self.get_file_loc(index), self.get_label(index)
        src_img = self.getImageData(file_loc)
        img = np.array(cv2.imdecode(np.asarray(bytearray(src_img), dtype=np.uint8), 1), dtype=np.float32)
        img /= 255.0
        img = img[:, :, ::-1]
        img = self.transforms(**{
            'image': img,
        })
        sample = {
            'input': img['image'],
            'label': label,
            'path': file_loc
        }
        #img = img.copy()

        return sample

    def __len__(self):
        return len(self.filelist)

    def getImageData(self, file_loc):
        data_name, tf_num, offset = file_loc
        tf_file = self.tfrecord_root  + "/" + data_name + "/" + data_name + "-" + tf_num + ".tfrecord"

        with open(tf_file, 'rb') as tf:
            tf.seek(int(offset))
            pb_len_bytes = tf.read(8)
            if len(pb_len_bytes) < 8:
                print("read pb_len_bytes err,len(pb_len_bytes)=" +
                      str(len(pb_len_bytes)))
                return None

            pb_len = struct.unpack('L', pb_len_bytes)[0]

            len_crc_bytes = tf.read(4)
            if len(len_crc_bytes) < 4:
                print("read len_crc_bytes err,len(len_crc_bytes)=" +
                      str(len(len_crc_bytes)))
                return None

            len_crc = struct.unpack('I', len_crc_bytes)[0]

            pb_data = tf.read(pb_len)
            if len(pb_data) < pb_len:
                print("read pb_data err,len(pb_data)=" + str(len(pb_data)))
                return None

            data_crc_bytes = tf.read(4)
            if len(data_crc_bytes) < 4:
                print("read data_crc_bytes err,len(data_crc_bytes)=" +
                      str(len(data_crc_bytes)))
                return None

            data_crc = struct.unpack('I', data_crc_bytes)[0]

            example = yt_example_pb2.Example()
            example.ParseFromString(pb_data)

            image_data_feature = example.features.feature.get("image")
            label_feature = example.features.feature.get("label")

            if image_data_feature:
                image_data = image_data_feature.bytes_list.value[0]
                return image_data

class TrainValDataset(dataset.Dataset):
    """ImageDataset for training.

    Args:
        datadir(str): dataset root path, default input and label dirs are 'input' and 'gt'
        aug(bool): data argument (×8)
        norm(bool): normalization

    Example:
        train_dataset = ImageDataset('train.txt', aug=False)
        for i, data in enumerate(train_dataset):
            input, label = data['input']. data['label']

    """

    def __init__(self, file_list, transforms, max_size=None):
        self.im_names = []
        self.labels = []
        with open(file_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip('\n')
                img, label = line.split(' ')
                label = int(label)
                self.im_names.append(img)
                self.labels.append(label)

        self.transforms = transforms
        self.max_size = max_size

    def __getitem__(self, index):
        """Get indexs by index

        Args:
            index(int): index

        Returns:
            {'input': input,
             'label': label,
             'path': path
            }

        """

        input = read_image(self.im_names[index])
        label = self.labels[index]

        sample = self.transforms(**{
            'image': input,
        })

        sample = {
            'input': sample['image'],
            'label': label,
            'path': self.im_names[index],
        }

        return sample

    def __len__(self):
        if self.max_size is not None:
            return min(self.max_size, len(self.im_names))

        return len(self.im_names)


class TestDataset(dataset.Dataset):
    """ImageDataset for test.

    Args:
        datadir(str): dataset path'
        norm(bool): normalization

    Example:
        test_dataset = ImageDataset('test', crop=256)
        for i, data in enumerate(test_dataset):
            input, file_name = data

    """

    def __init__(self, file_list, transforms, max_size=None):
        self.im_names = []
        with open(file_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip('\n')
                img = line
                self.im_names.append(img)

        self.transforms = transforms
        self.max_size = max_size


    def __getitem__(self, index):

        input = read_image(self.im_names[index])

        sample = self.transforms(**{
            'image': input,
        })

        sample = {
            'input': sample['image'],
            'path': self.im_names[index],
        }

        return sample

    def __len__(self):
        if self.max_size is not None:
            return min(self.max_size, len(self.im_names))

        return len(self.im_names)

