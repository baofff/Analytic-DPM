from torch.utils.data import Subset
from torchvision import datasets
import torchvision.transforms as transforms
from .dataset_factory import DatasetFactory
from .utils import *


class Cifar10(DatasetFactory):
    r""" Cifar10 dataset

    Information of the raw dataset:
         train: 40,000
         val:   10,000
         test:  10,000
         shape: 3 * 32 * 32
    """

    def __init__(self, data_path, gauss_noise=False, noise_std=0.01):
        super(Cifar10, self).__init__()
        self.data_path = data_path
        self.gauss_noise = gauss_noise
        self.noise_std = noise_std

        _transform = [transforms.ToTensor()]
        if self.gauss_noise:
            _transform.append(AddGaussNoise(self.noise_std))
        im_transform = transforms.Compose(_transform)
        self.train_val = datasets.CIFAR10(self.data_path, train=True, transform=im_transform, download=True)
        self.train = Subset(self.train_val, list(range(40000)))
        self.val = Subset(self.train_val, list(range(40000, 50000)))
        self.test = datasets.CIFAR10(self.data_path, train=False, transform=im_transform, download=True)

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
        return 3, 32, 32

    @property
    def fid_stat(self):
        return 'workspace/fid_stats/fid_stats_cifar10_train_pytorch.npz'
