from .utils import is_labelled, UnlabeledDataset
from torch.utils.data import ConcatDataset
import numpy as np


class DatasetFactory(object):
    r""" Output dataset after two transformations to the raw data:
    1. distribution transform (e.g. binarized, adding noise), often irreversible, a part of which is implemented
       in distribution_transform
    2. an affine transform (preprocess), which is bijective
    """

    def __init__(self):
        self.train = None
        self.val = None
        self.test = None

    def allow_labelled(self):
        return is_labelled(self.train)

    def get_data(self, dataset, labelled):
        assert not (not is_labelled(dataset) and labelled)
        if is_labelled(dataset) and not labelled:
            dataset = UnlabeledDataset(dataset)
        return self.affine_transform(self.distribution_transform(dataset))

    def get_train_data(self, labelled=False):
        return self.get_data(self.train, labelled=labelled)

    def get_val_data(self, labelled=False):
        return self.get_data(self.val, labelled=labelled)

    def get_train_val_data(self, labelled=False):
        train_val = ConcatDataset([self.train, self.val])
        return self.get_data(train_val, labelled=labelled)

    def get_test_data(self, labelled=False):
        return self.get_data(self.test, labelled=labelled)

    def distribution_transform(self, dataset):
        return dataset

    def affine_transform(self, dataset):
        return dataset

    def preprocess(self, v):
        r""" The mathematical form of the affine transform
        """
        return v

    def unpreprocess(self, v):
        r""" The mathematical form of the affine transform's inverse
        """
        return v

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None
