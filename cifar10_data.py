#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:24:02 2017

@author: Zizou

get data for cifar10
"""
import tensorflow as tf
import numpy as np
from data_generate import simulate_data, generation_net, variable_summaries
import tarfile
import requests
import os
from base_model import conv_net

def unpickle(file):
    """open a cpickle file and return a dict"""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_cifar_url(url):
    """
    Generate cifar10 data from origin source
    The archive contains the files data_batch_1, ..., data_batch_5, test_batch.
    Each of these files is a Python "pickled" object produced with cPickle

    Parameter
    ---------
    url: url for data download

    Returns
    -------
    X: cifar-10 data, shape: [60000, 3072]
    y: cifar-10 labels, shape: [60000, 10]

    """
    # download data
    print("data downloading...")
    r = requests.get(url)
    with open("cifar10.tar.gz", "wb") as code:
        code.write(r.content)
        print("data download finished")
    tar = tarfile.open("./cifar10.tar.gz")
    file_names = tar.getnames()
    # untargz data
    print("untargz data...")

    for file_name in file_names:
        tar.extract(file_name)
    tar.close()
    # remove tar.gz file
    os.remove("./cifar10.tar.gz")
    # read and merge data
    X = []
    y = []
    file_names = ["data_batch_1",
                  "data_batch_2",
                  "data_batch_3",
                  "data_batch_4",
                  "data_batch_5",
                  "test_batch"]
    for file_name in file_names:
        data_batch = unpickle("./cifar-10-batches-py/"+file_name)
        X.append(data_batch[b'data'])
        y.append(np.array(data_batch[b'labels']))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    print("merge data finished")
    return X, np.eye(10)[y]

def get_cifar():
    """
    Generate cifar10 data from origin source
    The archive contains the files data_batch_1, ..., data_batch_5, test_batch.
    Each of these files is a Python "pickled" object produced with cPickle

    Returns
    -------
    X: cifar-10 data, shape: [60000, 3072]
    y: cifar-10 labels, shape: [60000, 10]

    """
    # read and merge data
    X = []
    y = []
    file_names = ["data_batch_1",
                  "data_batch_2",
                  "data_batch_3",
                  "data_batch_4",
                  "data_batch_5",
                  "test_batch"]
    for file_name in file_names:
        data_batch = unpickle("./cifar-10-batches-py/"+file_name)
        X.append(data_batch[b'data'])
        y.append(np.array(data_batch[b'labels']))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    print("merge data finished")
    return X, np.eye(10)[y]

class conv_colored(conv_net):
    def __set_variable__(self, initializer, strides=[1,2,1,2]):
        """Set up network's variables"""
        full_size = int(self.n_input/(strides[1]**2)/(strides[3]**2))
        self.nn_layer([5,5,3], [32], "conv1", initializer)
        self.nn_layer([5,5,32], [64], "conv2", initializer)
        self.nn_layer([full_size*64], [1024], "fcon1", initializer)
        self.nn_layer([1024], [self.n_output], "output", initializer)


class generate_net_cifar(generation_net):
    def __init__(self, data, n_input = 3072, n_classes = 10, num_examples=60000):
        """
        Initialize network

        Parameters
        ----------
        data: object from a data class,
              have method mini-batch, attributes train, test, validation
        n_input: data input size(e.g. img shape: 32*32*3 = 3072)
        n_classes: total classes number(e.g. 0-9 digits; 0/1 labels)
        _index_in_epoch: index for next_train_batch method
        """
        super().__init__(data, n_input, n_classes, num_examples)

    def __construct_net__(self, strides = [1,2,1,2]):
        """
        Construct the structure of network.
        Record summaries to be shown in Tensorboard.

        Attributes
        ----------
        x: input data
        y: label data
        keep_prob: dropout
        learning_rate: learning rate
        net: generation net
        """
        with tf.variable_scope('input', reuse=None):
            self.x = tf.placeholder(tf.float32, shape=[None, self.n_input],name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes],name='y')

        with tf.variable_scope('hyperparameter', reuse=None):
            self.keep_prob = tf.placeholder(tf.float32, name='dropout')
            tf.summary.scalar('dropout_keep_probability', self.keep_prob)
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            tf.summary.scalar('learning_rate', self.learning_rate)

        with tf.variable_scope("generation_net_cifar", reuse=None):
            self.net = conv_colored("generation_net_cifar")
            self.net.network(x=self.x, n_input=self.n_input,
                             n_output=self.n_classes, dropout=self.keep_prob,
                             strides=strides, input_shape=[32,32,3],
                             initializer="he")
            with tf.variable_scope('output', reuse=None):
                variable_summaries(self.net.out, 'values')

def main():
    #url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    #X, y = get_cifar_url(url)
    X, y = get_cifar()
    data = simulate_data(shuffle=False, X=X, y=y)
    data.to_file(name="cifar10_origin", path="./simulation_data")

if __name__ == '__main__':
    main()
