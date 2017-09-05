#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 19:06:24 2017

@author: Zizou

Construct basic CNN model
"""

import tensorflow as tf
import numpy as np
from math import sqrt

def conv2d(x, W, b, strides=1):
    """
    Convolution operator 2D wrapper, with bias and relu activation.
    Use the padding = "SAME" will produce output with same size as input.

    Parameters
    ----------
    x: tensor, input data/output from last layer
    W: weight matrix for a small convolution mask
    b: bias matrix
    strides: strides, here we set columns stride = row stride

    Returns
    -------
    a tensor with conv and relu operator
    """
    # Conv2D wrapper, with bias and relu activation
    # input a*a, output a*a: padding = "SAME"
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    """
    MaxPool operator 2D wrapper.
    down-sampling procedure: input a*a, output a/k * a/k.

    Parameters
    ----------
    x: tensor, input data/output from last layer
    k: max_pool size and strides size

    Returns
    -------
    a down-sample tensor
    """
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def variable_summaries(var, name):
    """
    Attach a lot of summaries to a Tensor.
    Used in Tensorboard;
    Generate mean, min and max values.

    Parameters
    ----------
    var: tensor to be summarized
    name: tensor name/summary path
    """
    with tf.variable_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        #with tf.variable_scope('stddev'):
        #    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        #tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

def initial_variable(name, shape, dtype, initializer, mean=None, var=None):
    """
    Customized initialization methods

    Parameters
    ----------
    shape: list-like/array-like variable shape
    dtype: data type
    initializer: option: he, xavier, truncated(truncated normal),
                         constant(usually for biases), more to come...
    mean: set mean for truncated and constant methods
    var: set variance for truncated method

    Return
    ------
    a tensor: initialized variable with given options
    """
    if initializer == "xavier":
        return tf.get_variable(name=name, shape=shape, dtype=dtype,
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))

    elif initializer == "he":
        return tf.get_variable(name=name, shape=shape, dtype=dtype,
                               initializer=tf.contrib.layers.variance_scaling_initializer(uniform=False))

    elif initializer == "truncated":
        # mean and variance should be defined for truncated normal
        if not mean or not var:
            raise ValueError("Initializer error: mean and var should be defined for truncated normal.")
        initial = tf.truncated_normal(shape, mean=mean, stddev=0.01)
        return tf.Variable(initial, name=name, dtype=dtype)

    elif initializer == "constant":
        # mean should be defined for constant
        if not mean:
            raise ValueError("Initializer error: mean should be defined for constant.")
        initial = tf.constant(mean, shape=shape)
        return tf.Variable(initial, name=name, dtype=dtype)

    # one valid method should be chosen
    else:
        raise ValueError("Initializer error: choose a valid initializer")

class conv_net():
    """
    Convolution neural network for mean/variance estimation.

    Network Structure:
    1. Input data: 2D images k*k
    2. Conv1 layer: convolution layer with bias and ReLU op, total d1 mask;
                   output size: k*K*d1
    3. Maxpool layer: maxpool with stride size=k1 and ksize=k1;
                      output size: (k/k1)*(k/k1)*d1
    4. Conv2 layer: convolution layer with bias and ReLU op, total d2 mask;
                   output size: (k/k1)*(k/k1)*d2
    5. Maxpool layer: maxpool with stride size=k2 and ksize=k2;
                      output size: [k/(k1*k2)]*[k/(k1*k2)]*d2
    6. fully-connected1 layer: linear and ReLU op, dropout rate = dropout;
                               output size: 1024*1
    7. fully-connected output layer: linear op only, dropout rate = 1;
                                     output size: class_number*1(or class_number-1 * 1)
    """
    def __init__(self, name):
        """
        Initialize network.

        Parameters
        ----------
        name: network name

        Attributes
        ----------
        name: network name
        variables: variables dict of whole network
        """
        self.name = name
        self.variables = {}

    def nn_layer(self, input_dim, output_dim, layer_name, initializer="he"):
        """
        Reusable code for making a simple neural net layer.
        Construct with structure that can be shown in Tensorboard.

        Parameters
        ----------
        input_dim: input dimension, list type
        output_dim: output dimension, list type
        layer_name: layer name
        initializer: the choice of initialization method for layer weights and biases
        """
        # Calculate mean and variance in case of truncated_normal initialization
        var = 0.02/sqrt(np.prod(np.array(input_dim)))
        mean = var*2 + 1e-6
        with tf.variable_scope(layer_name, reuse=None):
            with tf.variable_scope("weight", reuse=None):
                self.variables[layer_name+'_w'] = initial_variable(name= 'weight',shape=input_dim+output_dim,
                                                                   dtype=tf.float32, initializer=initializer,
                                                                   mean=mean, var=var)


            with tf.variable_scope("bias", reuse=None):
                self.variables[layer_name+'_b'] = initial_variable(name= 'bias',shape=output_dim, dtype=tf.float32,
                                                                   initializer="constant", mean=mean)
        # Record variable summaries
        variable_summaries(self.variables[layer_name+'_w'], self.name+"/"+layer_name+'/weight')
        variable_summaries(self.variables[layer_name+'_b'], self.name+"/"+layer_name+'/bias')

    def __set_variable__(self, initializer, strides=[1,2,1,2]):
        """Set up network's variables"""

        full_size = int(self.n_input/(strides[1]**2)/(strides[3]**2))
        self.nn_layer([5,5,1], [32], "conv1", initializer)
        self.nn_layer([5,5,32], [64], "conv2", initializer)
        self.nn_layer([full_size*64], [1024], "fcon1", initializer)
        self.nn_layer([1024], [self.n_output], "output", initializer)

    def network(self, x, n_input, n_output, dropout, strides, initializer="he"):
        """
        Construct Convolution network.

        Parameters
        ----------
        x: tensors, input data
        n_input: input size
        n_output: output size
        dropout: dropout rate in fully-connected layer
        strides: numpy array with 4 elements
            1. conv layer 1 stride size
            2. max_pool layer 1 ksize and strides size
            3. conv layer 2 stride size
            4. max_pool layer 2 ksize and strides size
        initializer: initialization methods

        Attributes
        ----------
        n_input: input size
        n_output: output size
        out: output tensor

        Returns
        -------
        a output tensor
        """
        self.n_input = n_input
        self.n_output = n_output
        self.__set_variable__(strides=strides, initializer=initializer)

        x = tf.reshape(x, shape=[-1, int(np.sqrt(n_input)), int(np.sqrt(n_input)), 1])
        with tf.variable_scope('conv1', reuse=None):
            # Convolution Layer
            conv1 = conv2d(x, self.variables["conv1_w"], self.variables["conv1_b"], strides = strides[0])
            tf.summary.histogram(self.name+"/conv1/conv_output", conv1)
            # Max Pooling (down-sampling)
            conv1 = maxpool2d(conv1, k=strides[1])
            tf.summary.histogram(self.name+"/conv1/maxpool_output", conv1)
        with tf.variable_scope('conv2', reuse=None):
            # Convolution Layer
            conv2 = conv2d(conv1, self.variables["conv2_w"], self.variables["conv2_b"], strides = strides[2])
            tf.summary.histogram(self.name+"/conv2/conv_output", conv2)
            # Max Pooling (down-sampling)
            conv2 = maxpool2d(conv2, k=strides[3])
            tf.summary.histogram(self.name+"/conv2/maxpool_output", conv2)
        with tf.variable_scope('fcon1', reuse=None):
            # Fully connected layer
            # Reshape conv2 output to fit fully connected layer input
            fc1 = tf.reshape(conv2, [-1, self.variables["fcon1_w"].get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, self.variables["fcon1_w"]), self.variables["fcon1_b"])
            tf.summary.histogram(self.name+"/fcon1/fcon1_output", fc1)
            fc1 = tf.nn.relu(fc1)
            # Apply Dropout
            fc1 = tf.nn.dropout(fc1, dropout)
            tf.summary.histogram(self.name+"/fcon1/dropout_output", fc1)
        with tf.variable_scope('output', reuse=None):
            # Output
            self.out = tf.add(tf.matmul(fc1, self.variables["output_w"]), self.variables["output_b"])
            tf.summary.histogram(self.name+"/output/preact_output", self.out)

        return self.out

def label_modify(array, label_dict):
    """
    label modification method

    Parameters
    ----------
    array: np.array
    label_dict: mapping dictionary, with key the new label;
                                         value the list of old label
    """
    for key, value in label_dict.items():
        array[np.isin(array, value)] = key

def mnist_modify(data, label_dict, one_hot=False):
    """
    label modification method for mnist dataset

    Parameters
    ----------
    data: mnist dataset object
    label_dict: mapping dictionary, with key the new label;
                                         value the list of old label
    one_hot: whether use one_hot representation or not
    """
    # set attributes to write = True
    data.test.labels.setflags(write = 1)
    data.train.labels.setflags(write = 1)
    data.validation.labels.setflags(write=1)
    label_modify(data.test.labels, label_dict)
    label_modify(data.train.labels, label_dict)
    label_modify(data.validation.labels, label_dict)
    # modify labels for one_hot
    if one_hot == True:
        data.train._labels = np.eye(2)[data.train.labels]
        data.test._labels = np.eye(2)[data.test.labels]
        data.validation._labels = np.eye(2)[data.validation.labels]
