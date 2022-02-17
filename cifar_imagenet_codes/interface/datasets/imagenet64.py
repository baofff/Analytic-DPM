from PIL import Image
import os
import torchvision.transforms as transforms
from .dataset_factory import DatasetFactory
from .utils import *


class ImageDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        names = os.listdir(path)
        self.local_images = [os.path.join(path, name) for name in names]
        self._transform = transforms.ToTensor()

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        X = Image.open(self.local_images[idx])
        X = self._transform(X)
        return X


class Imagenet64(DatasetFactory):
    r""" Imagenet64 dataset

    Information of the raw dataset:
         train: 1,281,149
         test:  49,999
         shape: 3 * 64 * 64
    """

    def __init__(self, path):
        super().__init__()
        self.train = ImageDataset(os.path.join(path, 'train_64x64'))
        self.test = ImageDataset(os.path.join(path, 'valid_64x64'))

    def affine_transform(self, dataset):
        return StandardizedDataset(dataset, mean=0.5, std=0.5)  # scale to [-1, 1]

    def preprocess(self, v):
        return 2. * (v - 0.5)

    def unpreprocess(self, v):
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def data_shape(self):
        return 3, 64, 64

    @property
    def fid_stat(self):
        return 'workspace/fid_stats/fid_stats_imagenet64_train.npz'
