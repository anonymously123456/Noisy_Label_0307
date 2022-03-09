from logging import exception
from posixpath import join
import sys
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import copy
from tqdm import tqdm
import csv
from collections import Counter
import cv2
from scipy.special import comb
from numpy.testing import assert_array_almost_equal
import pandas as pd
from torchvision.datasets import ImageFolder
import PIL

np.seterr(divide='ignore', invalid='ignore')
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)


data_transforms = transforms.Compose([
    transforms.RandomRotation(90, resample=PIL.Image.BILINEAR),
    transforms.RandomApply((transforms.RandomHorizontalFlip(.5),
                            transforms.RandomVerticalFlip(.5)), p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """

    assert (noise >= 0.) and (noise <= 1.)

    P = noise / (size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (1 - noise) * np.ones(size))

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    y = y.numpy()
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return torch.from_numpy(new_y)


def noisify_with_P(y_train, nb_classes, noise, random_state=None):

    if noise > 0.0:
        P = build_uniform_P(nb_classes, noise)
        # seed the random numbers with #run
        y_train_noisy = multiclass_noisify(y_train,
                                           P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    else:
        P = np.eye(nb_classes)

    return y_train, P


# basic dataset class for load data
class BaiscDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.images = data['images']
        self.labels = data['labels']

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(transforms.ToPILImage()(image))

        return image, label


class CancerDataset(Dataset):

    def __init__(self, file_paths, labels, transform=None):
        super().__init__()
        self.file_paths = file_paths
        self.labels = labels
        # self.df = df_data.values
        # self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # img_name, label = self.df[index]
        label = self.labels[index]
        img_path = self.file_paths[index]
        try:
            # img_path = os.path.join(self.data_dir, img_name)
            image = cv2.imread(img_path)
            if self.transform is not None:
                image = self.transform(image)
        except Exception as e:
            print(img_path, e)
        return image, label


def PCam(data_path='../',
         dataset_list=['Noise'],
         noise_prob=0.1):
    NUM_CLASS = 2
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')

    all_pairs = pd.read_csv(os.path.join(data_path, 'hist_train_label.csv'))
    test_pairs = pd.read_csv(os.path.join(data_path, 'hist_val_label.csv'))

    all_pairs.head()
    train_pairs, val_pairs = train_test_split(
        all_pairs, stratify=all_pairs.label, test_size=0.1)
    train_file_list = train_pairs.values[:, 0]
    train_file_path = [os.path.join(train_path, i) for i in train_file_list]
    train_labels_int = train_pairs.values[:, 1].astype(np.int32)

    val_file_list = val_pairs.values[:, 0]
    val_labels_int = val_pairs.values[:, 1].astype(np.int32)
    val_file_path = [os.path.join(train_path, i) for i in val_file_list]

    test_file_path = [os.path.join(test_path, i)
                      for i in test_pairs.values[:, 0]]
    test_labels_int = test_pairs.values[:, 1].astype(np.int32)

    if dataset_list[0].lower() in 'noise':
        T = build_uniform_P(NUM_CLASS, noise_prob)
        print('T:', T)
        y_train_noisy = multiclass_noisify(
            torch.from_numpy(train_labels_int), P=T, random_state=0)

    return train_file_path, train_labels_int, val_file_path, val_labels_int, test_file_path, test_labels_int, y_train_noisy, T


def ISIC_dataset(data_path='../', dataset_list=['Noise'], noise_prob=0.1):
    NUM_CLASS = 2
    trainset = ImageFolder(os.path.join(
        data_path, 'train'), transform=data_transforms)
    testset = ImageFolder(os.path.join(data_path, 'test'),
                          transform=data_transforms)
    data_size = len(trainset)
    print(Counter(trainset.targets))
    validation_split = .2
    split = int(np.floor(validation_split * data_size))
    indices = list(range(data_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_labels = [trainset[i][1] for i in train_indices]
    train_path = [trainset.samples[i][0] for i in train_indices]
    if dataset_list[0].lower() in 'noise':
        T = build_uniform_P(NUM_CLASS, noise_prob)
        print('T:', T)
        y_train_noisy = multiclass_noisify(torch.from_numpy(
            np.array(train_labels)), P=T, random_state=0)
    assert len(train_path) == len(
        y_train_noisy), 'The number of data and labels should be matched'
    for k, v in enumerate(train_indices):
        assert trainset.samples[v][0] == train_path[k], 'The data path should not be changed'
        trainset.samples[v] = (train_path[k], y_train_noisy[k])
        trainset.imgs[v] = (train_path[k], y_train_noisy[k])
    return trainset, testset, T, train_sampler, val_sampler


def CIFAR10_fed(data_path='../data',
                dataset_list=['Noise'],
                noise_prob=0.1,
                downsample=True):

    NUM_CLASS = 10
    all_train_data = torchvision.datasets.CIFAR10(data_path,
                                                  train=True,
                                                  download=True,
                                                  transform=None)
    test_data = torchvision.datasets.CIFAR10(data_path,
                                             train=False,
                                             download=True,
                                             transform=None)

    test_data.targets = np.array(test_data.targets)
    test_data.data = torch.from_numpy(test_data.data).permute(0, 3, 1, 2)

    if downsample:
        idxs = dict()
        for i in range(10):
            idxs[i] = []
        for k, v in enumerate(all_train_data.targets):
            if len(idxs[v]) < 400:
                idxs[v].append(k)

        idx = []
        for i in range(10):
            idx.extend(idxs[i])

        random.shuffle(idx)
        all_train_data.data = torch.from_numpy(all_train_data.data[idx, ]).permute(
            0, 3, 1, 2)
        all_train_data.targets = torch.from_numpy(
            np.array(all_train_data.targets)[idx, ])

    # validation_data = all_train_data.data[40000:]
    # validation_targets = torch.tensor(all_train_data.targets[40000:])

    if downsample:
        validation_data = all_train_data.data[3200:]
        validation_targets = torch.tensor(all_train_data.targets[3200:])
        all_train_data.data = all_train_data.data[:3200]
        all_train_data.targets = torch.tensor(all_train_data.targets[:3200])
    else:
        validation_data = all_train_data.data[40000:]
        validation_targets = torch.tensor(all_train_data.targets[40000:])
        all_train_data.data = all_train_data.data[:40000]
        all_train_data.targets = torch.tensor(all_train_data.targets[:40000])

    # Data Corruption
    if dataset_list[0].lower() in 'noise':
        T = build_uniform_P(NUM_CLASS, noise_prob)
        y_train_noisy = multiclass_noisify(
            all_train_data.targets, P=T, random_state=0)

    train_data_dict = {
        'images': all_train_data.data,
        'labels': y_train_noisy
    }
    validation_data_dict = {
        'images': validation_data,
        'labels': validation_targets
    }
    test_data_dict = {
        'images': test_data.data,
        'labels': test_data.targets
    }
    return train_data_dict, validation_data_dict, test_data_dict, T
