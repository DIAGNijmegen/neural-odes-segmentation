import random

from augmentations import ElasticTransformations, RandomRotationWithMask

import cv2
import PIL
import torch
import numpy as np
import torchvision
import scipy.ndimage

cv2.setNumThreads(0)

class GLaSDataLoader(object):
    def __init__(self, patch_size, dataset_repeat=1, images=np.arange(0, 70), validation=False):
        self.image_fname = 'Warwick QU Dataset (Released 2016_07_08)/train_'
        self.images = images

        self.patch_size = patch_size
        self.repeat = dataset_repeat
        self.validation = validation

        self.image_mask_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            RandomRotationWithMask(45, resample=False, expand=False, center=None),
            ElasticTransformations(2000, 60),
            torchvision.transforms.ToTensor()
        ])
        self.image_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1),
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image, mask = self.index_to_filename(index)
        image, mask = self.open_and_resize(image, mask)
        image, mask = self.pad_image(image, mask)
        label, patch = self.apply_data_augmentation(image, mask)
        label = self.create_eroded_mask(label, mask)
        patch, label = self.extract_random_region(image, patch, label)
        return patch, label.float()

    def index_to_filename(self, index):
        """Helper function to retrieve filenames from index"""
        index_img = index // self.repeat
        index_img = self.images[index_img]
        index_str = str(index_img.item() + 1)

        image = self.image_fname + index_str + '.bmp'
        mask = self.image_fname + index_str + '_anno.bmp'
        return image, mask

    def open_and_resize(self, image, mask):
        """Helper function to pad smaller image to the correct size"""
        image = PIL.Image.open(image)
        mask = PIL.Image.open(mask)

        ratio = (775 / 512)
        new_size = (int(round(image.size[0] / ratio)),
                    int(round(image.size[1] / ratio)))

        image = image.resize(new_size)
        mask = mask.resize(new_size)

        image = np.array(image)
        mask = np.array(mask)
        return image, mask

    def pad_image(self, image, mask):
        """Helper function to pad smaller image to the correct size"""
        if not self.validation:
            pad_h = max(self.patch_size[0] - image.shape[0], 128)
            pad_w = max(self.patch_size[1] - image.shape[1], 128)
        else:
            # we pad more than needed to later do translation augmentation
            pad_h = max((self.patch_size[0] - image.shape[0]) // 2 + 1, 0)
            pad_w = max((self.patch_size[1] - image.shape[1]) // 2 + 1, 0)

        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        return padded_image, mask

    def apply_data_augmentation(self, image, mask):
        """Helper function to apply all configured data augmentations on both mask and image"""
        patch = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255
        n_glands = mask.max()
        label = torch.from_numpy(mask).float() / n_glands

        if not self.validation:
            patch_label_concat = torch.cat((patch, label[None, :, :].float()))
            patch_label_concat = self.image_mask_transforms(patch_label_concat)
            patch, label = patch_label_concat[0:3], np.round(patch_label_concat[3] * n_glands)
            patch = self.image_transforms(patch)
        else:
            label *= n_glands
        return label, patch

    def create_eroded_mask(self, label, mask):
        """Helper function to create a mask where every gland is eroded"""
        boundaries = torch.zeros(label.shape)
        for i in np.unique(mask):
            if i == 0: continue  # the first label is background
            gland_mask = (label == i).float()
            binarized_mask_border = scipy.ndimage.morphology.binary_erosion(gland_mask,
                                                                            structure=np.ones((13, 13)),
                                                                            border_value=1)

            binarized_mask_border = torch.from_numpy(binarized_mask_border.astype(np.float32))
            boundaries[label == i] = binarized_mask_border[label == i]

        label = (label > 0).float()
        label = torch.stack((boundaries, label))
        return label

    def extract_random_region(self, image, patch, label):
        """Helper function to perform translation data augmentation"""
        if not self.validation:
            loc_y = random.randint(0, image.shape[0] - self.patch_size[0])
            loc_x = random.randint(0, image.shape[1] - self.patch_size[1])
        else:
            loc_y, loc_x = 0, 0

        patch = patch[:, loc_y:loc_y+self.patch_size[0], loc_x:loc_x+self.patch_size[1]]
        label = label[:, loc_y:loc_y+self.patch_size[0], loc_x:loc_x+self.patch_size[1]]
        return patch, label

    def __len__(self):
        return len(self.images) * self.repeat
