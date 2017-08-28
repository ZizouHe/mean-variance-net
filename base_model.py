#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 19:06:24 2017

@author: Zizou

Construct basic CNN model
"""

import tensorflow as tf
import numpy as np

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
        izdict: initializer dict, currently includes Xavier's and He's initialization methods
        variables: variables dict of whole network
        """
        self.name = name
        self.izdict = {"xavier": tf.contrib.layers.xavier_initializer(uniform=True),
                        "He": tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN',
                                                                             uniform=False)}
        self.variables = {}

    def nn_layer(self, input_dim, output_dim, layer_name, initializer="xavier"):
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
        initializer = self.izdict[initializer]
        with tf.variable_scope(layer_name):
            with tf.variable_scope("weight"):
                self.variables[layer_name+'_w'] = tf.get_variable(name= 'weight',shape=input_dim+output_dim,
                                                                  dtype=tf.float32, initializer=initializer)


            with tf.variable_scope("bias"):
                self.variables[layer_name+'_b'] = tf.get_variable(name= 'bias',shape=output_dim,
                                                                  dtype=tf.float32, initializer=initializer)
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

    def network(self, x, n_input, n_output, dropout, strides, initializer="xavier"):
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
        initializer: the choice of initialization method

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
        self.__set_variable__(initializer=initializer, strides=strides)

        x = tf.reshape(x, shape=[-1, int(np.sqrt(n_input)), int(np.sqrt(n_input)), 1])
        with tf.variable_scope('conv1'):
            # Convolution Layer
            conv1 = conv2d(x, self.variables["conv1_w"], self.variables["conv1_b"], strides = strides[0])
            tf.summary.histogram(self.name+"/conv1/conv_output", conv1)
            # Max Pooling (down-sampling)
            conv1 = maxpool2d(conv1, k=strides[1])
            tf.summary.histogram(self.name+"/conv1/maxpool_output", conv1)
        with tf.variable_scope('conv2'):
            # Convolution Layer
            conv2 = conv2d(conv1, self.variables["conv2_w"], self.variables["conv2_b"], strides = strides[2])
            tf.summary.histogram(self.name+"/conv2/conv_output", conv2)
            # Max Pooling (down-sampling)
            conv2 = maxpool2d(conv2, k=strides[3])
            tf.summary.histogram(self.name+"/conv2/maxpool_output", conv2)
        with tf.variable_scope('fcon1'):
            # Fully connected layer
            # Reshape conv2 output to fit fully connected layer input
            fc1 = tf.reshape(conv2, [-1, self.variables["fcon1_w"].get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, self.variables["fcon1_w"]), self.variables["fcon1_b"])
            tf.summary.histogram(self.name+"/fcon1/fcon1_output", fc1)
            fc1 = tf.nn.relu(fc1)
            # Apply Dropout
            fc1 = tf.nn.dropout(fc1, dropout)
            tf.summary.histogram(self.name+"/fcon1/dropout_output", fc1)
        with tf.variable_scope('output'):
            # Output
            self.out = tf.add(tf.matmul(fc1, self.variables["output_w"]), self.variables["output_b"])
            tf.summary.histogram(self.name+"/output/preact_output", self.out)

        return self.out
