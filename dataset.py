from pathlib import Path

import numpy as np

import cv2
import re
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, base_directory, material_filter='ALU', test=False, test_split=0.1):
        super().__init__()
        self.test = test
        self.materials = {'ALU': 0, 'STEEL': 1}
        self.grain_sizes = {'40': 0, '80': 1, '120': 2,
                            '180': 3, '240': 4, '400': 5, '500': 6}
        base_directory = Path(base_directory)
        folder_directories = [folder_directory for folder_directory in base_directory.iterdir(
        ) if folder_directory.is_dir()]

        self.train_set = []
        self.test_set = []
        for folder_directory in folder_directories:
            image_paths = sorted(
                [image_path for image_path in folder_directory.iterdir() if image_path.is_file()])
            material, grain_size = re.match(
                r'(\w+)_p(\d+)', folder_directory.name).group(1, 2)
            if material != material_filter:
                continue
            for index, image_path in enumerate(image_paths):
                image = {'image_path': image_path,
                         'material': self.materials[material], 'grain_size': self.grain_sizes[grain_size]}
                if index < len(image_paths) * test_split:
                    self.test_set.append(image)
                else:
                    self.train_set.append(image)

    def __len__(self):
        if self.test:
            return len(self.test_set)
        else:
            return len(self.train_set)

    def __repr__(self):
        return f'Dataset with {len(self)} images'

    def __getitem__(self, index):
        if self.test:
            image = self.test_set[index]
        else:
            image = self.train_set[index]
        image['image'] = cv2.imread(image['image_path'].as_posix())
        image.pop('image_path', None)
        return image
