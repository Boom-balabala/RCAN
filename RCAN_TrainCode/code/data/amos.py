import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class AMOS(srdata.SRData):#继承srdata类
    def __init__(self, args, train=True):
        super(AMOS, self).__init__(args, train)
        self.batch_size = args.batch_size
    def _scan(self):# 生成高低分辨率的list
        list_hr = []
        # [[]]
        list_lr = [[] for _ in self.scale] # 这里scale不知道是什么
        if self.train:
            idx_begin = 0
            # n_train=800 DI2K的从867张图片作为训练集合
            idx_end = self.args.n_train
        else:
            idx_begin = self.args.n_train
            # 测试n_train--self.args.offset_val（800） + self.args.n_val（5）
            idx_end = self.args.offset_val + self.args.n_val

        for i in range(idx_begin + 1, idx_end + 1):
            # "0001" "0002"
            filename = '{:0>4}'.format(i)
            # hr=['C:\\Users\\17581\\Desktop\\RCAN-master\\RCAN-master/DIV2K\\DIV2K_train_HR\\0001.png']
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            for si, s in enumerate(self.scale):# s=4,si=0
                list_lr[si].append(os.path.join(# [['C:\\Users\\17581\\Desktop\\RCAN-master\\RCAN-master/DIV2K\\DIV2K_train_LR_bicubic\\X4/0001x4.png']]
                    self.dir_lr,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                ))

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        # 'C:\\Users\\17581\\Desktop\\RCAN-master\\RCAN-master/AMOS'
        self.apath = dir_data + '/AMOS'
        # "'C:\\Users\\17581\\Desktop\\RCAN-master\\RCAN-master/AMOS\\DIV2K_train_HR'"
        self.dir_hr = os.path.join(self.apath, 'train_HR')
        # 'C:\\Users\\17581\\Desktop\\RCAN-master\\RCAN-master/DIV2K\\DIV2K_train_LR_bicubic'
        self.dir_lr = os.path.join(self.apath, 'train_LR')
        self.ext = '.png'

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )

    def __len__(self):
        if self.train:
            return ((len(self.images_hr) * 20)// self.batch_size) *self.batch_size
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

