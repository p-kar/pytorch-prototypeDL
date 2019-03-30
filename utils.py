import os
import pdb
import time
import random
import shutil
import numpy as np

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

import PIL
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, sigma, alpha):
    '''
    this code is borrowed from chsasank on GitHubGist
    Elastic deformation of images as described in [Simard 2003].
    
    image: a three-dimensional numpy array representing the PIL image
    sigma: the real-valued variance of the gaussian kernel
    alpha: a real-value that is multiplied onto the displacement fields
    
    returns: an elastically distorted image of the same shape
    '''
    assert len(image.shape) == 3
    # the two lines below ensure we do not alter the array images
    e_image = np.empty_like(image)
    e_image[:] = image
    height = image.shape[0]
    width = image.shape[1]
    
    random_state = np.random.RandomState(None)
    x, y = np.mgrid[0:height, 0:width]
    dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode='constant') * alpha
    dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode='constant') * alpha
    indices = x + dx, y + dy
    
    for i in range(e_image.shape[2]):
        e_image[:, :, i] = map_coordinates(e_image[:, :, i], indices, order=1)

    return e_image

class MNISTElasticTranform(object):
    """
    Applies Elastic Transform on MNIST images
    """

    def __init__(self, sigma, alpha):
        self.sigma = sigma
        self.alpha = alpha

    def __call__(self, sample):
        image = sample
        image = np.asarray(image).reshape(28, 28, 1)
        image = elastic_transform(image, self.sigma, self.alpha)
        image = PIL.Image.fromarray(image.reshape(28, 28))

        return image

class CIFAR10ElasticTransform(object):
    """
    Applies Elastic Transform on CIFAR images
    """

    def __init__(self, sigma, alpha):
        self.sigma = sigma
        self.alpha = alpha

    def __call__(self, sample):
        image = sample
        image = np.asarray(image)
        image = elastic_transform(image, self.sigma, self.alpha)
        image = PIL.Image.fromarray(image)

        return image

def get_mnist_loaders(data_dir, bsize, num_workers, sigma, alpha):
    transform = transforms.Compose([MNISTElasticTranform(sigma, alpha), transforms.ToTensor()])

    train_set = datasets.MNIST(root=data_dir, train=True, download=True,
        transform=transform)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=bsize, num_workers=num_workers)

    valid_set = datasets.MNIST(root=data_dir, train=False, download=True,
        transform=transforms.ToTensor())
    valid_loader = DataLoader(valid_set, shuffle=False, batch_size=bsize, num_workers=num_workers)

    return train_loader, valid_loader

def get_cifar_loaders(data_dir, bsize, num_workers, sigma, alpha):
    transform = transforms.Compose([CIFAR10ElasticTransform(sigma, alpha), transforms.ToTensor()])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=bsize, shuffle=True, num_workers=num_workers)

    valid_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
    valid_loader = DataLoader(valid_set, batch_size=bsize, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader

def get_fmnist_loaders(data_dir, bsize, num_workers, sigma, alpha):
    transform = transforms.Compose([MNISTElasticTranform(sigma, alpha), transforms.ToTensor()])

    train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True,
        transform=transform)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=bsize, num_workers=num_workers)

    valid_set = datasets.FashionMNIST(root=data_dir, train=False, download=True,
        transform=transforms.ToTensor())
    valid_loader = DataLoader(valid_set, shuffle=False, batch_size=bsize, num_workers=num_workers)

    return train_loader, valid_loader
