#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:03:31 2017

@author: Zizou
"""

"""
Construct MNIST Data
all functions for reading and manipulating data are included in mndata class
"""
from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import scipy
from sklearn.utils import resample

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

class mndata():
    def __init__(self, one_hot = False, scale = True):
        """
        Construcing the MNIST dataset.
        one_hot arg is used if necessary,
        scale arg is True if pixel range is 0~255.
        """
        # load data from original MNIST dataset
        self._data = MNIST('./MNIST_data/')
        self._classnum = NUM_CLASSES
        self._one_hot = False
        self._relabel = False
        self._X_train, self._labels_train = map(np.array, self._data.load_training())
        self._X_test, self._labels_test = map(np.array, self._data.load_testing())
        if scale:
            self._X_train = self._X_train/255.0
            self._X_test = self._X_test/255.0

        if one_hot:
            self.one_hot()
        self._scale = scale
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = self._X_train.shape[0]

    def one_hot(self):
        """
        Change from original scalar-form labels to vector-form one-hot labels
        """
        if self._one_hot is True:
            print("the dataset is already one-hot labeled")
            return
        self.__one_hot__(train = True)
        self.__one_hot__(train = False)
        self._one_hot = True
        print("One hot finished")

    def __one_hot__(self, train = True):
        if train:
            self._labels_train =  np.eye(self._classnum)[self._labels_train]
        else:
            self._labels_test =  np.eye(self._classnum)[self._labels_test]

    def relabel(self, classnum, classlist):
        """
        Change the data labels from 0~9 to any user-given classes
        classlist is used to denote the new classes
        """
        self._classnum = classnum
        self._classlist = classlist
        self.__relabel__(train = True)
        self.__relabel__(train = False)
        self._relabel = True
        print("Relabeling finished")

    def __relabel__(self, train = True):
        for i, lst in enumerate(self._classlist):
            for label in lst:
                if train:
                    self._labels_train[self._labels_train == label] = i
                else:
                    self._labels_test[self._labels_test == label] = i

    def next_batch(self, batch_size):
        """
        Return the next `batch_size` examples from this data set.
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # if finish the epoch
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            # shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._X_train = self._X_train[perm]
            self._labels_train = self._labels_train[perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._X_train[start:end], self._labels_train[start:end]

    def divide(self, num = 50):
        """
        Divide data into several parts
        """
        self._division = np.arange(self._num_examples)
        np.random.shuffle(self._division)
        self._divnum = num

    def divde_next_batch(self, batch_size, divrank = 1):
        """
        Mini batch on divided part
        divrank arg ranges from 1 to division number
        """
        if divrank < self._divnum:
            n = self._num_examples//self._divnum
            perm = self._division[(divrank - 1)*n : divrank*n]
        elif divrank == self._divnum:
            n = self._num_examples - (self._num_examples//self._divnum) \
                                    *(self._divnum - 1)
            perm = self._division[(divrank - 1)*n :]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # if finish the epoch
        if self._index_in_epoch > n:
            self._epochs_completed += 1
            # shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._X_train = self._X_train[perm]
            self._labels_train = self._labels_train[perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._X_train[perm][start:end], self._labels_train[perm][start:end]

    def resample(self):
        """
        Resample the data, using bootstrap method
        """
        # get the raw data
        self._X_train, self._labels_train = map(np.array, self._data.load_training())
        # Scale the data again
        if self._scale:
            self._X_train = self._X_train/255.0
        if self._relabel:
            self.__relabel__(train = True)
        if self._one_hot:
            self.__one_hot__(train = True)
        # resample
        self._X_train, self._labels_train = resample(self._X_train, self._labels_train)


    @property
    def X_train(self):
        return self._X_train

    @property
    def X_test(self):
        return self._X_test

    @property
    def labels_train(self):
        return self._labels_train

    @property
    def labels_test(self):
        return self._labels_test


