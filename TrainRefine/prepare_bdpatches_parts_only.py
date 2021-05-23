import sys

sys.path.append("../")

import argparse
import torch
from torchvision import transforms
from preprocess import Stage1ToTensor, Resize, ToPILImage
from torch.utils.data import DataLoader
from Datasets.Helen import HelenDataset
from utils.augmentation import Stage1Augmentation
from prefetch_generator import BackgroundGenerator
from utils.calc_funcs import get_dets, affine_crop
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image

# Dataset Read_in Part

root_dir = "../datas/data"
parts_root_dir = "../datas/parts"

txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt",
    'test': "testing.txt"
}

transforms_list = {
    'train':
        transforms.Compose([
            ToPILImage(),
            Resize((512, 512)),
            Stage1ToTensor()
        ]),
    'val':
        transforms.Compose([
            ToPILImage(),
            Resize((512, 512)),
            Stage1ToTensor()
        ]),
    'test':
        transforms.Compose([
            Stage1ToTensor()
        ])
}


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# DataLoader
Dataset = {x: HelenDataset(txt_file=txt_file_names[x],
                           root_dir=root_dir,
                           parts_root_dir=parts_root_dir,
                           transform=transforms_list[x],
                           stage='stage1'
                           )
           for x in ['train', 'val', 'test']
           }
#
# stage1_augmentation = Stage1Augmentation(dataset=HelenDataset,
#                                          txt_file=txt_file_names,
#                                          root_dir=root_dir,
#                                          parts_root_dir=parts_root_dir,
#                                          resize=(512, 512),
#                                          stage='stage1'
#                                          )
# enhaced_stage1_datasets = stage1_augmentation.get_dataset()

dataloader = {x: DataLoaderX(Dataset[x], batch_size=1,
                             shuffle=False, num_workers=10)
              for x in ['train', 'val', 'test']
              }
device = torch.device('cuda:1')
plt_show = False
parts_name = ['lbrow', 'rbrow', 'leye', 'reye', 'nose', 'mouth']
# prepare parts bd
cropped_parts_bd_out_path = "/bigdata/yinzi/patchset/parts"
for x in parts_name:
    os.makedirs(os.path.join(cropped_parts_bd_out_path, 'train', x), exist_ok=True)
    os.makedirs(os.path.join(cropped_parts_bd_out_path, 'val', x), exist_ok=True)
    os.makedirs(os.path.join(cropped_parts_bd_out_path, 'test', x), exist_ok=True)
# patches 16
size_set = [8, 8, 8, 8, 16, 16]

for x in ['train', 'val', 'test']:
    for batch in tqdm(dataloader[x]):
        image = batch['image'].to(device)
        label = batch['labels'].to(device)
        name = batch['name']
        N = image.shape[0]
        # image Shape [1, 3, 512, 512]
        # label Shape [1, 9, 512, 512]
        label = torch.cat([label[:, 1:6], label[:, 6:9].sum(dim=1, keepdim=True)], dim=1)
        # Shape (1, 6, 512, 512)
        for i in range(6):
            boxes = get_dets(label[:, i:i + 1], patch_size=size_set[i], out_fmt='cxcywh')
            boxes_xyxy = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
            image_patches, theta = affine_crop(image, boxes[0], is_label=False)
            label_patches, _ = affine_crop(label[:, i:i + 1], theta=theta, is_label=True)
            if plt_show:
                pathces_grid = torchvision.utils.make_grid(image_patches).detach().cpu()
                label_pathces_grid = torchvision.utils.make_grid(label_patches).detach().cpu()
                show_boxes = torchvision.utils.draw_bounding_boxes(
                    torch.from_numpy(np.array(TF.to_pil_image(image[0]))).permute(2, 0, 1),
                    boxes_xyxy[0])
                plt.imshow(TF.to_pil_image(show_boxes))
                plt.pause(0.01)
                plt.imshow(pathces_grid.permute(1, 2, 0))
                plt.pause(0.01)
                plt.imshow(label_pathces_grid[0])
                plt.pause(0.01)
            for j in range(len(image_patches)):
                TF.to_pil_image(image_patches[j]).save(os.path.join(cropped_parts_bd_out_path, x, parts_name[i],
                                                                    f'{name[0]}_image_{j}.png'), format="PNG",
                                                       compress_level=0)
                TF.to_pil_image(label_patches[j]).save(os.path.join(cropped_parts_bd_out_path, x, parts_name[i],
                                                                    f'{name[0]}_label_{j}.png'), format="PNG",
                                                       compress_level=0)
        # torch.save(theta.detach().cpu(), os.path.join(cropped_parts_bd_out_path, x, f"{name}_theta.pth"))
# prepare skin bd

# prepare hair bd
