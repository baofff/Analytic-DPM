import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10
from .celeba import CelebA
# from .ffhq import FFHQ
from .lsun import LSUN
from torch.utils.data import Subset
import numpy as np
from torch.utils.data import Dataset


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class StandardizedDataset(Dataset):
    def __init__(self, dataset, mean, std):
        self.dataset = dataset
        self.mean = mean
        self.std = std
        self.std_inv = 1. / std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x = self.dataset[item]
        return self.std_inv * (x - self.mean)


def get_dataset(dataset):

    if dataset == "celeba":
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64

        dataset = CelebA(
            root="workspace/datasets/celeba/",
            split="train",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )

        test_dataset = CelebA(
            root=os.path.join("workspace/datasets/celeba/"),
            split="test",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(64),
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )

    elif dataset == "lsun_bedroom":
        # if config.data.random_flip:
        #     dataset = LSUN(
        #         root="workspace/datasets/lsun_bedroom",
        #         classes=["bedroom_train"],
        #         transform=transforms.Compose(
        #             [
        #                 transforms.Resize(256),
        #                 transforms.CenterCrop(256),
        #                 transforms.RandomHorizontalFlip(p=0.5),
        #                 transforms.ToTensor(),
        #             ]
        #         ),
        #     )
        # else:
        dataset = LSUN(
            root="workspace/datasets/lsun_bedroom",
            classes=["bedroom_train"],
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                ]
            ),
        )

        test_dataset = None

        # test_dataset = LSUN(
        #     root="workspace/datasets/lsun_bedroom",
        #     classes=["bedroom_val"],
        #     transform=transforms.Compose(
        #         [
        #             transforms.Resize(256),
        #             transforms.CenterCrop(256),
        #             transforms.ToTensor(),
        #         ]
        #     ),
        # )

    elif dataset == "lsun_church":
        dataset = LSUN(
            root="workspace/datasets/lsun_church",
            classes=["church_outdoor_train"],
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                ]
            ),
        )
        test_dataset = None

    elif dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(
                path=os.path.join(args.exp, "datasets", "FFHQ"),
                transform=transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()]
                ),
                resolution=config.data.image_size,
            )
        else:
            dataset = FFHQ(
                path=os.path.join(args.exp, "datasets", "FFHQ"),
                transform=transforms.ToTensor(),
                resolution=config.data.image_size,
            )

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = (
            indices[: int(num_items * 0.9)],
            indices[int(num_items * 0.9) :],
        )
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
    else:
        dataset, test_dataset = None, None

    return StandardizedDataset(dataset, mean=0.5, std=0.5), StandardizedDataset(test_dataset, mean=0.5, std=0.5)


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
