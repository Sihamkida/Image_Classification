import torch
import functools

import numpy as np
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DatasetCachingWrapper(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return f'{self.dataset} in CachingWrapper'

    def __getitem__(self, index):
        return self._getitem(index)

    @functools.lru_cache(maxsize=None)
    def _getitem(self, index):
        return self.dataset[index]


class DatasetIndexingWrapper(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, indices):
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return f'{self.dataset} in IndexingWrapper with {len(self.indices)} items'

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


class ClassificationMetricCalculator:
    def __init__(self):
        super().__init__()
        self._reset()

    def add(self, prediction, gt):
        result = torch.argmax(prediction, dim=1) == gt
        self.num_samples += result.shape[0]
        self.num_positives += result[result == True].shape[0]
        self.num_negatives += result[result == False].shape[0]

    def finish(self):
        accuracy = self.num_positives / self.num_samples
        self._reset()
        return accuracy

    def _reset(self):
        self.num_samples = 0
        self.num_positives = 0
        self.num_negatives = 0


def prepare_input(batch, train=True):
    if train:
        transform = transforms.Compose([
            transforms.CenterCrop(512),
            transforms.RandomCrop(256),
            # transforms.RandomRotation(180),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            # transforms.GaussianBlur(5),
            # transforms.RandomAdjustSharpness(2),
            # transforms.RandomAutocontrast(),
            # transforms.RandomEqualize(),
            # transforms.Normalize([0.55618479, 0.51933078, 0.49571777], [0.1273316, 0.11998756, 0.12270675]), # ALU only
            # transforms.Normalize([0.53842587, 0.51321856, 0.49315634], [0.12934684, 0.12307481, 0.1252288]), # STEEL only
            # transforms.Normalize([0.54748573, 0.51633675, 0.49446307], [0.12862943, 0.12154804, 0.12395518]), # full dataset
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(256),
            # transforms.Normalize([0.55618479, 0.51933078, 0.49571777], [0.1273316, 0.11998756, 0.12270675]), # ALU only
            # transforms.Normalize([0.53842587, 0.51321856, 0.49315634], [0.12934684, 0.12307481, 0.1252288]), # STEEL only
            # transforms.Normalize([0.54748573, 0.51633675, 0.49446307], [0.12862943, 0.12154804, 0.12395518]), # full dataset
        ])       
    images = []
    for image in batch['image']:
        image = image.permute(2, 0, 1)
        image = image / 255
        image = transform(image).float()
        image = (image - image.min()) / (image.max() - image.min())
        images.append(image)
    input = torch.stack(images).to(device)
    gt = {'material': batch['material'].to(
        device), 'grain_size': batch['grain_size'].to(device)}
    return input, gt


def crop(image, position='random', height=256, width=256, border=20):
    full_width, full_height, _ = image.shape  # original image size

    if position == 'center':
        left = round(full_width / 2 - width / 2)
        top = round(full_height / 2 - height / 2)
        right = left + width
        bottom = top + height
        return image[left:right, top:bottom, :]

    elif position == 'random':
        center = [np.random.randint(round(width / 2) + border, full_width - round(width / 2) - border, dtype=int),
                  np.random.randint(round(height / 2) + border, full_height - round(height / 2) - border, dtype=int)]
        left = center[0] - round(width / 2)
        right = left + width
        top = center[1] - round(height / 2)
        bottom = top + height
        return image[left:right, top:bottom, :]

    else:
        raise ValueError('Invalid position: Choose "random" or "center".')


def mirror(image):
    return np.fliplr(image)

def flip(image):
    return np.flipud(image)

def flipMirror(image):
    return flip(mirror(image))

def meanStd(dataset):
    red = np.zeros(len(dataset))
    green = np.zeros(len(dataset))
    blue = np.zeros(len(dataset))
    red_std = np.zeros(len(dataset))
    green_std = np.zeros(len(dataset))
    blue_std = np.zeros(len(dataset))

    i = 0
    for dataset in dataset:
        image = dataset['image']
        red[i] = np.mean(image[:,:,0])
        green[i] = np.mean(image[:,:,1])
        blue[i] = np.mean(image[:,:,2])
        red_std[i] = np.std(image[:,:,0])
        green_std[i] = np.std(image[:,:,1])
        blue_std[i] = np.std(image[:,:,2])
        i += 1

    rgb_mean = [np.mean(red),
                np.mean(green),
                np.mean(blue)]
    rgb_std = [np.std(red_std),
                np.std(green_std),
                np.std(blue_std)]
    return rgb_mean, rgb_std