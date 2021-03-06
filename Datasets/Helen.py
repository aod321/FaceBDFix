import numpy as np
import os
from torch.utils.data import Dataset
from skimage import io
import cv2
import torch


class HelenDataset(Dataset):
    # HelenDataset

    def __init__(self, txt_file, root_dir, parts_root_dir, stage=None, transform=None, only_parts=True,
                 train_subset=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = 'train'
        if txt_file == "exemplars.txt":
            self.mode = 'train'
        elif txt_file == "testing.txt":
            self.mode = 'test'
        elif txt_file == "tuning.txt":
            self.mode = 'val'
        if train_subset and self.mode == 'train':
            self.name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype="str", delimiter=',')[:train_subset]
        else:
            self.name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype="str", delimiter=',')
        self.only_parts = only_parts
        self.root_dir = root_dir
        self.parts_root_dir = parts_root_dir
        self.transform = transform
        self.stage = stage

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        img_path = os.path.join(self.root_dir, 'images',
                                img_name + '.jpg')
        labels_path = [os.path.join(self.root_dir, 'labels',
                                    img_name,
                                    img_name + "_lbl%.2d.png") % i
                       for i in range(11)]

        image = io.imread(img_path)
        image = np.array(image)
        labels = [io.imread(labels_path[i]) for i in range(11)]
        labels = np.array(labels)
        if self.only_parts:
            # bg = labels[0] + labels[1] + labels[10]
            bg = 255 - labels[2:10].sum(0)
            labels = np.uint8(np.concatenate(([bg.clip(0, 255)], labels[2:10]), axis=0))
        else:
            bg = 255 - labels[1:11].sum(0)
            labels = np.uint8(np.concatenate(([bg.clip(0, 255)], labels[1:11]), axis=0))


        if self.stage == 'stage1':
            sample = {'image': image, 'labels': labels, 'name': img_name}
        else:
            parts, parts_mask = self.getparts(idx)
            orig_size = image.shape
            sample = {'image': image, 'labels': labels, 'orig': image, 'orig_label': labels, 'orig_size': orig_size,
                      'parts_gt': parts, 'parts_mask_gt': parts_mask, 'name': img_name}

        if self.transform:
            sample = self.transform(sample)
            new_label = sample['labels']
            new_label_fg = torch.sum(new_label[1:], dim=0, keepdim=True)  # 1 x 128 x 128
            new_label[0] = 1. - new_label_fg
            sample['labels'] = new_label
        return sample

    def getparts(self, idx):
        name = self.name_list[idx, 1].strip()
        name_list = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']
        path = {x: os.path.join(self.parts_root_dir, x, self.mode)
                for x in name_list}
        parts_path = {x: os.path.join(path[x], name + "_image.png")
                      for x in name_list}
        parts_mask_path = {x: os.path.join(path[x], name + "_label.png")
                           for x in name_list}
        parts = [io.imread(parts_path[x])
                 for x in name_list]

        parts_mask = [cv2.imread(parts_mask_path[x], cv2.IMREAD_GRAYSCALE).astype(np.float32())
                      for x in name_list]  # (H, W)

        return parts, parts_mask


class SkinHelenDataset(Dataset):
    # HelenDataset

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype="str", delimiter=',')
        self.mode = 'train'
        if txt_file == "exemplars.txt":
            self.mode = 'train'
        elif txt_file == "testing.txt":
            self.mode = 'test'
        elif txt_file == "tuning.txt":
            self.mode = 'val'
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        img_path = os.path.join(self.root_dir, 'images',
                                img_name + '.jpg')
        labels_path = [os.path.join(self.root_dir, 'labels',
                                    img_name,
                                    img_name + "_lbl%.2d.png") % i
                       for i in range(11)]

        image = io.imread(img_path)
        image = np.array(image)
        labels = [io.imread(labels_path[i]) for i in range(11)]
        labels = np.array(labels)
        bg = 255 - labels[1:11].sum(0)
        labels = np.uint8(np.concatenate(([bg.clip(0, 255)], labels[1:11]), axis=0))
        sample = {'image': image, 'labels': labels, 'name': img_name}

        if self.transform:
            sample = self.transform(sample)
            new_label = sample['labels']
            new_label_fg = torch.sum(new_label[1:], dim=0, keepdim=True)  # 1 x 128 x 128
            new_label[0] = 1. - new_label_fg
            sample['labels'] = new_label
        return sample

    def getparts(self, idx):
        name = self.name_list[idx, 1].strip()
        name_list = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']
        path = {x: os.path.join(self.parts_root_dir, x, self.mode)
                for x in name_list}
        parts_path = {x: os.path.join(path[x], name + "_image.png")
                      for x in name_list}
        parts_mask_path = {x: os.path.join(path[x], name + "_label.png")
                           for x in name_list}
        parts = [io.imread(parts_path[x])
                 for x in name_list]

        parts_mask = [cv2.imread(parts_mask_path[x], cv2.IMREAD_GRAYSCALE).astype(np.float32())
                      for x in name_list]  # (H, W)

        return parts, parts_mask


class PartsDataset(Dataset):

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype="str", delimiter=',')
        self.mode = 'train'
        if txt_file == "exemplars.txt":
            self.mode = 'train'
        elif txt_file == "testing.txt":
            self.mode = 'test'
        elif txt_file == "tuning.txt":
            self.mode = 'val'
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        name_list = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']
        path = {x: os.path.join(self.root_dir, x, self.mode)
                for x in name_list}
        parts_path = {x: os.path.join(path[x], img_name + "_image.png")
                      for x in name_list}
        parts_mask_path = {x: os.path.join(path[x], img_name + "_label.png")
                           for x in name_list}
        parts = [io.imread(parts_path[x])
                 for x in name_list]

        parts_mask = [cv2.imread(parts_mask_path[x], cv2.IMREAD_GRAYSCALE).astype(np.float32())
                      for x in name_list]  # (H, W)

        sample = {'image': parts, 'labels': parts_mask}

        if self.transform:
            sample = self.transform(sample)
        return sample
