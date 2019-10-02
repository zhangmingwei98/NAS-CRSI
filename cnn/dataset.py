import os
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import glob
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

rgb_mean = (0.4353, 0.4452, 0.4131)
rgb_std = (0.2044, 0.1924, 0.2013)

class MyDataset(Dataset):
    def __init__(self,
                 config,
                 args,
                 subset):
        super(MyDataset, self).__init__()
        assert subset == 'train' or subset == 'val' or subset == 'test'

        self.args = args
        self.config = config
        self.root = args.input
        self.subset = subset
        self.data = self.config.data_folder_name # image
        self.target = self.config.target_folder_name # label

        #self.data_transforms = data_transforms if data_transforms!=None else TF.to_tensor
        #self.target_transforms = target_transforms if target_transforms!= None else TF.to_tensor

        self.mapping = {
            0: 0,
            255: 1,
        }
        #讲原图和标记数据列表读入
        self.data_list = glob.glob(os.path.join(
            self.root,
            subset,
            self.data,
            '*'
        ))
        self.target_list = glob.glob(os.path.join(
            self.root,
            subset,
            self.target,
            '*'
        ))


    def mask_to_class(self, mask):
        for k in self.mapping:
            mask[mask == k] = self.mapping[k]
        return mask

    def train_transforms(self, image, mask):

        #将短边resize到input_size
        resize = transforms.Resize(size=(self.config.input_size, self.config.input_size))
        image = resize(image)
        mask = resize(mask)

        #随机水平翻转图像
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        #随机垂直翻转图像
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        #将图像转为tensor，并进行归一化处理
        image = TF.to_tensor(image) # scale 0-1
        image = TF.normalize(image, mean=rgb_mean, std=rgb_std) # normalize

        #从
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        mask = self.mask_to_class(mask)
        mask = mask.long()
        return image, mask

    def untrain_transforms(self, image, mask):

        resize = transforms.Resize(size=(self.config.input_size, self.config.input_size))
        image = resize(image)
        mask = resize(mask)

        # 没有旋转的变化

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=rgb_mean, std=rgb_std)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        mask = self.mask_to_class(mask)
        mask = mask.long()
        return image, mask

    #实例化对象P，可以用P[key]进行取值
    def __getitem__(self, index):

        datas = Image.open(self.data_list[index])
        targets = Image.open(self.target_list[index])
        if self.subset == 'train':
            t_datas, t_targets = self.train_transforms(datas, targets)
            return t_datas, t_targets
        elif self.subset == 'val':
            t_datas, t_targets = self.untrain_transforms(datas, targets)
            return t_datas, t_targets
        elif self.subset == 'test':
            t_datas, t_targets = self.untrain_transforms(datas, targets)
            return t_datas, t_targets, self.data_list[index]

    #返回数据的数量
    def __len__(self):

        return len(self.data_list)





