import numpy as np
import os
from torch.utils.data import Dataset
from skimage import io
import cv2
import torch
from glob import glob
from natsort import natsorted
import re
from PIL import Image


class RefineDataSet(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        super(RefineDataSet, self).__init__()
        path = os.path.join(root_dir, mode)
        print(path)
        # if mode == 'trian':
        #     self.img_path_list = natsorted(glob(os.path.join(path, "*/*_image_*.png")))
        #     self.label_path_list = natsorted(glob(os.path.join(path, "*/*_label_*.png")))
        #     self.rough_path_list = natsorted(glob(os.path.join(path, "*/*_rough_*.png")))
        # elif mode == 'test':
        #     self.img_path_list = natsorted(glob(os.path.join(path, "*/*_image_*.png")))
        #     self.label_path_list = natsorted(glob(os.path.join(path, "*/*_label_*.png")))
        #     self.rough_path_list = natsorted(glob(os.path.join(path, "*/*_rough_*.png")))
        # else:
        self.img_path_list = natsorted(glob(os.path.join(path, "*/*_image_*.png")))
        self.label_path_list = natsorted(glob(os.path.join(path, "*/*_label_*.png")))
        self.rough_path_list = natsorted(glob(os.path.join(path, "*/*_rough_*.png")))

        self.transform = transform
        self.mode = mode

    def path2name(self, path):
        name = re.findall(".*/(.*)_.*_.*.png", path)
        if len(name) != 0:
            return name[0]
        else:
            return None

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        image, label = Image.open(self.img_path_list[idx]), Image.open(self.label_path_list[idx])
        rough_mask = Image.open(self.rough_path_list[idx])
        sample = {'image': image, 'labels': label, 'rough_mask': rough_mask}
        if self.transform:
            sample = self.transform(sample)
        return sample
