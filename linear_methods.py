#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:21:36 2017

@author: Zizou

Some traditional methods on alpha-beta structure
"""

import random
import sklearn.metrics as metrics
import numpy as np
import scipy
from tensorflow.examples.tutorials.mnist import input_data
from base_model import mnist_modify

NUM_CLASSES = 2
dim = 784
d = 4000
w = np.random.normal(0, 0.1, (dim, d))
b = np.random.uniform(0, 2*np.pi, d)

def phi(X):
    ''' Multiply the 784-dimensional MNIST vectors by unit normal '''
    return np.cos(X.dot(w) + b)

class linear_regression():
    def __init__(self, data):
        self.data = data
        self.param_alpha = np.random.normal(loc, scale, size)

    def cost_func(self, alpha, beta, x, y):
        """cost function"""
        out = scipy.special.beta(alpha+y[:,0], beta+y[:,1])/scipy.special.beta(alpha, beta)
        return np.log(out).mean() +

    def train(self, Lambda, learning_rate, training_iter, display_step, batch_size):
        """training procedure"""
        self._Lambda = Lambda






if __name__ == "__main__":
    # read and munipulate data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    # change labels 0~9 to {0,1}
    mnist_modify(mnist, {0:range(0,5), 1:range(5,10)}, one_hot=True)
    # project 784-d vector onto 4000-d space
    mnist.test.images.setflags(write = 1)
    mnist.train.images.setflags(write = 1)
    mnist.validation.images.setflags(write=1)
    mnist.train._images = phi(mnist.train.images)
    mnist.test._images = phi(mnist.test.images)
    mnist.validation._images = phi(mnist.validation.images)
