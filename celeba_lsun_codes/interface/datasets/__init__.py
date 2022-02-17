from .cifar10 import Cifar10
from .other_dst import get_dataset


def get_test_dataset(name):
    if name == 'cifar10':
        return Cifar10("workspace/datasets/cifar10/").get_test_data(False)
    elif name == 'celeba':
        return get_dataset('celeba')[1]


def get_train_dataset(name):
    if name == 'cifar10':
        return Cifar10("workspace/datasets/cifar10/").get_train_val_data(False)
    elif name == 'celeba':
        return get_dataset('celeba')[0]
    elif name == 'lsun_bedroom':
        return get_dataset('lsun_bedroom')[0]
    elif name == 'lsun_church':
        return get_dataset('lsun_church')[0]
    else:
        raise NotImplementedError
