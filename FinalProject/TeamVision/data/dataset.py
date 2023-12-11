

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

CLASSIFICATION = ('background',  # always index 0
                  'aeroplane',
                  'bicycle',
                  'bird',
                  'boat',
                  'bottle',
                  'bus',
                  'car',
                  'cat',
                  'chair',
                  'cow',
                  'diningtable',
                  'dog',
                  'horse',
                  'motorbike',
                  'person',
                  'pottedplant',
                  'sheep',
                  'sofa',
                  'train',
                  'tvmonitor')

NUM_CLASSES = len(CLASSIFICATION) + 1


class PascalVOCDataset(Dataset):
    def __init__(self, list_file, img_dir, mask_dir, transform=None):
        self.images = open(list_file, "rt").read().split("\n")[:-1]
        self.transform = transform

        self.img_extension = ".jpg"
        self.mask_extension = ".png"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir

        self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(self.image_root_dir, name + self.img_extension)
        mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

        image = self.load_image(path=image_path)
        gt_mask = self.load_mask(path=mask_path)

        data = {
            'image': torch.FloatTensor(image),
            'mask': torch.LongTensor(gt_mask)
        }

        return data

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))

        for name in self.images:
            mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

            raw_image = Image.open(mask_path).resize((224, 224))
            imx_t = np.array(raw_image).reshape(224 * 224)
            imx_t[imx_t == 255] = len(CLASSIFICATION)

            for i in range(NUM_CLASSES):
                counts[i] += np.sum(imx_t == i)

        return counts

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values / np.sum(values)

        return torch.Tensor(p_values)

    def load_image(self, path=None):
        raw_image = Image.open(path)
        raw_image = np.transpose(raw_image.resize((224, 224)), (2, 1, 0))
        imx_t = np.array(raw_image, dtype=np.float32) / 255.0

        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize((224, 224))
        imx_t = np.array(raw_image)
        # border
        imx_t[imx_t == 255] = len(CLASSIFICATION)

        return imx_t
