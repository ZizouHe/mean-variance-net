#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:21:36 2017

@author: Zizou

Some traditional methods on alpha-beta structure
"""

import numpy as np
import scipy
from tensorflow.examples.tutorials.mnist import input_data
from base_model import mnist_modify
import copy

NUM_CLASSES = 2
dim = 784
d = 4000
w = np.random.normal(0, 0.1, (dim, d))
b = np.random.uniform(0, 2*np.pi, d)

def project(data):
    ''' Multiply the 784-dimensional MNIST vectors by unit normal '''
    def phi(X):
        return np.cos(X.dot(w) + b)
    data.train._images = phi(data.train.images)
    data.test._images = phi(data.test.images)
    data.validation._images = phi(data.validation.images)
    print("data projection finished")

class linear_regression():
    """Linear Regression"""
    def __init__(self, data):
        """
        linear regression initialization method

        Parameters
        ----------
        data: class simulate_data or tensorflow.examples.tutorials.mnist.input_data

        Attributes
        ----------
        _data: data set with train, test, validation set
        _n_input: input size
        _n_classes: number of classes
        _alpha_weights: alpha matrix with size [self._n_input+1, self._n_classes-1]
        _beta_weights: beta matrix with size [self._n_input+1, self._n_classes-1]
        """
        self._data = copy.deepcopy(data)
        self._n_input = data.train.images.shape[1]
        self._n_classes = data.train.labels.shape[1]
        # add constant 1 on input data
        self.__add_one__()
        # initialize weights
        self._alpha_weights = np.random.normal(loc=0, scale=1, size=(self._n_input,
                                                                     self._n_classes-1))
        #print(self._alpha_weights.shape)
        self._beta_weights = np.random.normal(loc=0, scale=1, size=(self._n_input,
                                                                    self._n_classes-1))
        # add biases
        zeros = np.zeros((1, self._n_classes-1))
        self._alpha_weights = np.concatenate((self._alpha_weights,zeros), axis=0)
        self._beta_weights = np.concatenate((self._beta_weights,zeros), axis=0)

    def __add_one__(self):
        """add constant 1 on input data"""
        ones = np.ones((self._data.train.images.shape[0],1))
        self._data.train._images = np.concatenate((self._data.train._images, ones), axis=1)
        ones = np.ones((self._data.train.images.shape[0],1))
        self._data.test._images = np.concatenate((self._data.train._images, ones), axis=1)
        ones = np.ones((self._data.train.images.shape[0],1))
        self._data.validation._images = np.concatenate((self._data.train._images, ones), axis=1)

    def cost_func(self, X, y):
        """
        cost function
        ！Warning: currently only for two classes

        Parameters
        ----------
        X: input, numpy matrix
        y: labels, numpy matrix/array

        Returns
        -------
        cost function value on this batch
        """
        alpha = X.dot(self._alpha_weights)
        beta = X.dot(self._beta_weights)
        out = scipy.special.beta(alpha+y[:,0], beta+y[:,1])/scipy.special.beta(alpha, beta)
        # regularization without biases
        norm = self._Lambda/2*(np.sum(self._alpha_weights[:self._n_input,:]**2)+
                               np.sum(self._beta_weights[:self._n_input,:]**2))
        return np.log(out).mean() + norm

    def SGD(self, X, y):
        alpha = X.dot(self._alpha_weights)
        beta = X.dot(self._beta_weights)



    def train(self, Lambda, learning_rate, training_iter, display_step, batch_size):
        """training procedure"""
        self._Lambda = Lambda




if __name__ == "__main__":
    # read and manipulate data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    # change labels 0~9 to {0,1}
    mnist_modify(mnist, {0:range(0,5), 1:range(5,10)}, one_hot=True)
    # project 784-d vector onto 4000-d space
    project(mnist)

