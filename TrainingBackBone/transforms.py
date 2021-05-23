from torchvision import transforms
from torchvision.transforms import functional as TF
import numpy as np
from skimage.util import random_noise
import torch
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F


class RandomAffine(transforms.RandomAffine):
    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        fill = self.fill
        img, labels = sample['image'], sample['labels']
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * TF._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        img_size = TF._get_image_size(img)

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

        new_img = TF.affine(img, *ret, interpolation=self.interpolation, fill=fill)
        new_labels = TF.affine(labels, *ret, interpolation=self.interpolation, fill=fill)
        sample.update({'image': new_img, 'labels': new_labels})
        return sample


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        img, labels = sample['image'], sample['labels']
        if torch.rand(1) < self.p:
            img = TF.hflip(img)
            labels = TF.hflip(labels)
        sample.update({'image': img, 'labels': labels})
        return sample


class ToTensor(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

         Override the __call__ of transforms.ToTensor
    """

    def __call__(self, sample):
        """
                Args:
                    dict of pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

                Returns:y
                    Tensor: Converted image.
        """
        image, labels = sample['image'], sample['labels']
        sample.update({'image': TF.to_tensor(image), 'labels': TF.to_tensor(labels)})
        return sample


class ToPILImage(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

         Override the __call__ of transforms.ToTensor
    """

    def __call__(self, sample):
        """
                Args:
                    dict of pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

                Returns:y
                    Tensor: Converted image.
        """
        image, labels = sample['image'], sample['labels']
        sample.update({'image': TF.to_pil_image(image), 'labels': TF.to_pil_image(labels)})
        return sample


class GaussianNoise(object):
    def __call__(self, sample):
        img = sample['image']
        img = np.where(img != 0, random_noise(img), img)
        sample.update({'image': img})
        return sample


class Normalize(transforms.Normalize):
    def __call__(self, sample):
        img = sample['image']
        TF.normalize(img, self.mean, self.std, self.inplace)
        sample.update({'image': img})
        return sample


class RandomResizedCrop(transforms.RandomResizedCrop):
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = TF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        label = TF.resized_crop(label, i, j, h, w, self.size, InterpolationMode.NEAREST)
        sample.update({'image': img, 'label': label})
        return sample


class Resize(transforms.Resize):
    """Resize the input PIL Image to the given size.
             Override the __call__ of transforms.Resize
    """

    def __call__(self, sample):
        """
            Args:
                 sample:{'image':PIL Image to be resized,'labels':labels to be resized}

             Returns:
                 sample:{'image':resized PIL Image,'labels': resized PIL label list}

        """
        image, labels = sample['image'], sample['labels']
        resized_image = TF.to_pil_image(F.interpolate(TF.to_tensor(image).unsqueeze(0),
                                                      self.size, mode='bilinear', align_corners=True).squeeze(0)
                                        )

        resized_labels = TF.resize(labels, self.size, InterpolationMode.NEAREST)

        # assert resized_labels.shape == (9, 128, 128)
        sample.update({'image': resized_image, 'labels': resized_labels})

        return sample
