#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 00:49:36 2017

@author: Zizou
"""

import tensorflow as tf
import numpy as np
from math import sqrt
from base_model import variable_summaries
import copy
from cifar10_data import get_cifar
from data_generate import simulate_data
import time

IMAGE_SIZE = 24
NUM_CLASSES = 10
INPUT_SIZE = 3072
INITIAL_LEARNING_RATE = 0.01
TRAINING_ITERS = 6000
DISPLAY_STEP = 20
BATCH_SIZE = 128

def conv2d(x, W, b, strides=1):
    """
    Convolution operator 2D wrapper,
    with bias and relu activation.
    Use the padding = "SAME" will produce
    output with same size as input.

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
    x = tf.nn.conv2d(
        x,
        W,
        strides=[1, strides, strides, 1],
        padding='SAME')

    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2, name=None):
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
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME',
        name=name)

def image_processing(X):
    """
    image munipulation

    Parameters
    ----------
    1D tensor or 1D numpy array

    Returns
    -------
    3D tensor, resized and reshaped image
    """
    # reshape inmage to a 3D tensor of [32, 32, 3]
    reshaped_image = tf.reshape(tf.cast(X, tf.float32), shape=[32,32,3])

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(
        reshaped_image,
        height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    return float_image

def initial_variable(name, shape, dtype, initializer, var=None):
    """
    Customized initialization methods

    Parameters
    ----------
    shape: list-like/array-like variable shape
    dtype: data type
    initializer: option: he, xavier, truncated(truncated normal),
                         constant(usually for biases), more to come...
    var: set variance for truncated method

    Return
    ------
    a tensor: initialized variable with given options
    """
    if initializer == "xavier":
        return tf.get_variable(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer(uniform=True))

    elif initializer == "he":
        return tf.get_variable(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=tf.contrib.layers.variance_scaling_initializer(uniform=False))

    elif initializer == "truncated":
        # variance should be defined for truncated normal
        if not var:
            raise ValueError("Initializer error: \
                var should be defined for truncated normal.")

        return tf.get_variable(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=tf.truncated_normal_initializer(stddev=var))

    elif initializer == "constant":
        initial = tf.constant(0.0, shape=shape, dtype=dtype)
        return tf.Variable(initial, name=name, dtype=dtype)

    # one valid method should be chosen
    else:
        raise ValueError("Initializer error: choose a valid initializer")


class CNN():
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
        #var = 4/sqrt(np.prod(np.array(input_dim)))
        var = 5e-2
        with tf.variable_scope(layer_name, reuse=None):
            with tf.variable_scope("weight", reuse=None):
                self.variables[layer_name+'_w'] = initial_variable(
                    name= 'weight',
                    shape=input_dim+output_dim,
                    dtype=tf.float32,
                    initializer=initializer,
                    var=var)


            with tf.variable_scope("bias", reuse=None):
                self.variables[layer_name+'_b'] = initial_variable(
                    name= 'bias',
                    shape=output_dim,
                    dtype=tf.float32,
                    initializer="constant")
        # Record variable summaries
        variable_summaries(
            self.variables[layer_name+'_w'],
            self.name+"/"+layer_name+'/weight')
        variable_summaries(
            self.variables[layer_name+'_b'],
            self.name+"/"+layer_name+'/bias')

    def __set_variable__(self, initializer, strides=[1,2,1,2]):
        """Set up network's variables"""

        full_size = int(IMAGE_SIZE*IMAGE_SIZE/(strides[1]**2)/(strides[3]**2))
        self.nn_layer([5,5,3], [64], "conv1", initializer)
        self.nn_layer([5,5,64], [64], "conv2", initializer)
        self.nn_layer([full_size*64], [384], "local3", initializer)
        self.nn_layer([384], [192], "local4", initializer)
        self.nn_layer([192], [NUM_CLASSES], "fcon5", initializer)

    def network(self, x, strides,initializer="truncated"):
        """
        Construct Convolution network.

        Parameters
        ----------
        x: tensors, input data
        strides: numpy array with 4 elements
            1. conv layer 1 stride size
            2. max_pool layer 1 ksize and strides size
            3. conv layer 2 stride size
            4. max_pool layer 2 ksize and strides sizee
        initializer: initialization methods

        Returns
        -------
        a output tensor
        """
        self.__set_variable__(
            strides=strides,
            initializer=initializer)

        x = tf.map_fn(image_processing, x, dtype=tf.float32)
        with tf.variable_scope('conv1', reuse=None):
            # Convolution Layer
            conv1 = conv2d(
                x,
                self.variables["conv1_w"],
                self.variables["conv1_b"],
                strides = strides[0])
            tf.summary.histogram(
                self.name+"/conv1/conv_output",
                conv1)
            # Max Pooling (down-sampling)
            conv1 = maxpool2d(
                conv1,
                k=strides[1],
                name='pool1')
            tf.summary.histogram(
                self.name+"/conv1/maxpool_output",
                conv1)

        # norm1
        norm1 = tf.nn.lrn(conv1, 4,
                            bias=1.0, alpha=0.001 / 9.0,
                            beta=0.75, name='norm1')

        with tf.variable_scope('conv2', reuse=None):
            # Convolution Layer
            conv2 = conv2d(
                conv1,
                self.variables["conv2_w"],
                self.variables["conv2_b"],
                strides = strides[2])
            tf.summary.histogram(
                self.name+"/conv2/conv_output",
                conv2)
            # Max Pooling (down-sampling)
            conv2 = maxpool2d(
                conv2,
                k=strides[3],
                name='pool2')
            tf.summary.histogram(
                self.name+"/conv2/maxpool_output",
                conv2)

        norm2 = tf.nn.lrn(conv2, 4,
                            bias=1.0, alpha=0.001 / 9.0,
                            beta=0.75, name='norm2')


        with tf.variable_scope('local3', reuse=None):
            # Local Connected Layer
            local3 = tf.reshape(
                conv2,
                [-1, self.variables["local3_w"].get_shape().as_list()[0]])
            local3 = tf.add(
                tf.matmul(local3, self.variables["local3_w"]),
                self.variables["local3_b"])
            tf.summary.histogram(
                self.name+"/local3/local_output",
                local3)
            local3 = tf.nn.relu(local3)
            tf.summary.histogram(
                self.name+"/local3/activation_output",
                local3)

        with tf.variable_scope('local4', reuse=None):
            # Local Connected Layer
            local4 = tf.add(
                tf.matmul(local3, self.variables["local4_w"]),
                self.variables["local4_b"])
            tf.summary.histogram(
                self.name+"/local4/local_output",
                local4)
            local4 = tf.nn.relu(local4)
            tf.summary.histogram(
                self.name+"/local4/activation_output",
                local4)

        with tf.variable_scope('fcon5', reuse=None):
            # Output
            self.logits = tf.add(
                tf.matmul(local4, self.variables["fcon5_w"]),
                self.variables["fcon5_b"])
            tf.summary.histogram(
                self.name+"/fcon5/preact_output",
                self.logits)

        return self.logits

class generation_net():
    def __init__(self, data):
        self.data = copy.deepcopy(data)

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
            self.x = tf.placeholder(
                tf.float32,
                shape=[None, INPUT_SIZE],
                name='x')
            self.y = tf.placeholder(
                tf.float32,
                shape=[None, NUM_CLASSES],
                name='y')

        with tf.variable_scope('hyperparameter', reuse=None):
            """
            self.learning_rate = tf.placeholder(
                tf.float32,
                name='learning_rate')
            tf.summary.scalar('learning_rate', self.learning_rate)
            """
            self.global_steps = tf.Variable(
                0,
                dtype=tf.int32,
                name='globel_steps')

        with tf.variable_scope("generation_net", reuse=None):
            self.net = CNN("generation_net")
            self.net.network(
                x=self.x,
                strides=strides,
                initializer="truncated")
            with tf.variable_scope('output', reuse=None):
                variable_summaries(self.net.logits, 'values')

    def __define_measure__(self):
        """
        Define cost and accuracy in mean var net.
        Record summaries to be shown in Tensorboard.

        Attributes
        ----------
        accuracy: classification accuracy
        cost: cost function on training set
        """
        def l2_loss(weights, wd):
            """
            l2 loss

            Parameters
            ----------
            weights: tf.variables tensor,
                weights to be regularized
            wd: lambda

            Returns
            -------
            a loss tensor
            """
            wd = tf.constant(wd, tf.float32)
            return tf.multiply(wd, tf.nn.l2_loss(weights))

        with tf.variable_scope('accuracy', reuse=None):
            with tf.variable_scope('correct_prediction', reuse=None):
                correct_pred = tf.equal(
                    tf.argmax(self.net.logits,1),
                    tf.argmax(self.y,1))
            with tf.variable_scope('accuracy', reuse=None):
                self.accuracy = tf.reduce_mean(
                    tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.variable_scope('cost', reuse=None):
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.net.logits,
                    labels=self.y))
            self.cost = tf.add(
                self.cost,
                tf.add(
                    l2_loss(self.net.variables['local3_w'], wd=0.004),
                    l2_loss(self.net.variables['local4_w'], wd=0.004)))
        tf.summary.scalar('train_cost', self.cost)

    def __optimization__(self):
        """Define optimization methods for networks."""
        with tf.variable_scope('optimization', reuse=None):
            opt = tf.train.AdamOptimizer(
                learning_rate=INITIAL_LEARNING_RATE)
            self.optimizer = opt.minimize(
                loss=self.cost,
                global_step=self.global_steps)

    def train_net(self, strides = [1,2,1,2]):
        """
        Train conv nets: data generation DNN.

        Parameters
        ----------
        training_iters: number of iterations during training
        learning_rate: learning rate
        batch_size: data size of mini-batch
        display_step: how many steps gap to show training result
        dropout: Dropout, probability to keep units
        strides: list with 4 elements
            1. conv layer 1 stride size for conv1 layer
            2. max_pool layer 1 ksize and strides size for conv1 layer
            3. conv layer 2 stride size for conv2 layer
            4. max_pool layer 2 ksize and strides size for conv2 layer
        """
        # Initializing network
        self.__construct_net__(strides=strides)
        self.__define_measure__()
        self.__optimization__()
        init = tf.global_variables_initializer()
        merged = tf.summary.merge_all()

        # Launch the graph
        with tf.Session() as sess:
            print("\nTraining start...")
            sess.run(init)
            step = 1
            # Record train data
            # $ tensorboard --logdir=./tensorboard_cifar10
            train_writer = tf.summary.FileWriter(
                './tensorboard_cifar10',
                sess.graph)
            start_time = current_time = time.time()
            # Keep training until reach max iterations
            while step * BATCH_SIZE <= TRAINING_ITERS:
                batch_x, batch_y = self.data.train.next_batch(BATCH_SIZE)
                # Run optimization op (backprop)
                summary, _ = sess.run([merged, self.optimizer],
                                       feed_dict={self.x: batch_x,
                                                  self.y: batch_y})
                train_writer.add_summary(summary, step)
                # display training intermediate result
                if step % DISPLAY_STEP == 0:
                    t = time.time()
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([self.cost,self.accuracy],
                                          feed_dict={self.x: batch_x,
                                                     self.y: batch_y})
                    print("Iter " + str(step*BATCH_SIZE) +\
                        ", Time Cost: {:.2f}s".format(t - current_time) +\
                        ". Loss= " + "{:.3f}".format(loss) +\
                        ", Accuracy= " + "{:.2f}".format(acc))
                    current_time = t

                step += 1

            end_time = time.time()
            print("Optimization finished. Total training time: %.2fs"\
                %(end_time - start_time))
            # Calculate test loss
            #self.pred, loss, acc = sess.run([self.net.logits, self.cost, self.accuracy],
            #                                 feed_dict={self.x: self.data.train.images,
            #                                            self.y: self.data.train.labels})
            self.pred, loss, acc = sess.run([self.net.logits, self.cost, self.accuracy],
                                             feed_dict={self.x: batch_x,
                                                        self.y: batch_y})
            print("Train Loss= {:.4f}".format(loss) +", Train Accuracy= {:.4f}".format(acc))

def main():
    #url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    #X, y = get_cifar_url(url)
    X, y = get_cifar()
    data = simulate_data(shuffle=False, X=X, y=y)
    data.merge_all()
    cnn = generation_net(data=data)
    del data
    cnn.train_net(strides=[1,2,1,2])

if __name__ == '__main__':
    main()
