#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:50:48 2017

@author: Zizou
Construct alpha-beta net
"""

from base_model import conv_net, variable_summaries, mnist_modify
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def cost_func(alpha, beta, y):
    """
    Cost function for alpha-beta net.

    Parameters
    ----------
    alpha: a tensor with size [sample_size, n_classes-1], output from alpha net
    beta: a tensor with size [sample_size, n_classes-1], output from beta net
    y: a tensor with size [sample_size, n_classes], labels

    Return
    ------
    A scaler tensor, cost
    """
    def beta_func(vector):
        """
        Calculate value of beta function
        Parameter
        ---------
        a vector tensor of size [1,2], indicates [alpha, beta] input

        Return
        ------
        a scaler tensor, the value of beta function
        """
        nomi = tf.multiply(tf.exp(tf.lgamma(vector[0])), tf.exp(tf.lgamma(vector[1])))
        deno = tf.exp(tf.lgamma(tf.add(vector[0],vector[1])))
        return tf.div(nomi,deno)

    output = tf.concat([alpha, beta],1)
    cost = tf.div(tf.map_fn(beta_func, tf.add(output, y)),
                  tf.map_fn(beta_func, output))
    return -tf.reduce_mean(tf.log(cost))

class alpha_beta_net():
    """Alpha-Beta Network for Classification."""
    def __init__(self, data, n_input = 784, n_classes = 2):
        """
        Initialize network

        Parameters
        ----------
        data: object from a data class,
              have method mini-batch, attributes train, test, validation
        n_input: data input size(e.g. img shape: 28*28 = 784)
        n_classes: total classes number(e.g. 0-9 digits; 0/1 labels)
        """
        self.data = data
        self.n_input = n_input
        self.n_classes = n_classes

    def __construct_net__(self, strides=[1,2,1,2,1,2,1,2]):
        """
        Construct the structure of network.
        Record summaries to be shown in Tensorboard.

        Attributes
        ----------
        x: input data
        y: label data
        keep_prob: dropout
        learning_rate: learning rate
        alpha_net: mean network
        beta_net: var network
        """
        with tf.variable_scope('input', reuse=None):
            self.x = tf.placeholder(tf.float32, shape=[None, self.n_input],name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes],name='y')

        with tf.variable_scope('hyperparameter', reuse=None):
            self.keep_prob = tf.placeholder(tf.float32, name='dropout')
            tf.summary.scalar('dropout_keep_probability', self.keep_prob)
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            tf.summary.scalar('learning_rate', self.learning_rate)

        with tf.variable_scope('alpha_net', reuse=None):
            self.alpha_net = conv_net('alpha_net')
            # construct alpha network use truncated-normal initializer
            self.alpha_net.network(x=self.x, n_input=self.n_input,
                                   n_output=self.n_classes-1, dropout=self.keep_prob,
                                   strides=strides[:4], initializer="truncated")
            self.alpha_net.out = tf.nn.elu(self.alpha_net.out)+1

            with tf.variable_scope('output', reuse=None):
                #self.alpha_net.out = tf.sigmoid(self.alpha_net.out)
                variable_summaries(self.alpha_net.out, 'values')

        with tf.variable_scope('beta_net', reuse=None):
            self.beta_net = conv_net('beta_net')
            # construct alpha network use truncated-normal initializer
            self.beta_net.network(x=self.x, n_input=self.n_input,
                                   n_output=self.n_classes-1, dropout=self.keep_prob,
                                   strides=strides[4:], initializer="truncated")
            self.beta_net.out = tf.nn.elu(self.beta_net.out)+1

            with tf.variable_scope('output', reuse=None):
                #self.beta_net.out = tf.sigmoid(self.beta_net.out)
                variable_summaries(self.beta_net.out, 'values')

    def __define_measure__(self):
        """
        Define cost and accuracy in mean var net.
        Record summaries to be shown in Tensorboard.

        Attributes
        ----------
        accuracy: classification accuracy
        cost: cost function on training set
        validation_cost: cost function value on validation set
        """
        with tf.variable_scope('accuracy', reuse=None):
            with tf.variable_scope('correct_prediction', reuse=None):
                correct_pred = tf.equal(tf.argmax(tf.concat([self.alpha_net.out,
                                                             self.beta_net.out],1),1),
                                        tf.argmax(self.y,1))
            with tf.variable_scope('accuracy', reuse=None):
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.variable_scope('cost', reuse=None):
            self.cost = cost_func(self.alpha_net.out,self.beta_net.out,self.y)
            self.validation_cost = cost_func(self.alpha_net.out,self.beta_net.out,self.y)
        tf.summary.scalar('train_cost', self.cost)
        tf.summary.scalar('validation_cost', self.validation_cost)

    def __optimization__(self, learning_rate=0.001):
        """Define optimization methods for networks."""
        with tf.variable_scope('optimization'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def train_net(self, training_iters = 20000, learning_rate = 0.001, batch_size = 100,
                  display_step = 1, dropout = 0.75, strides = [1,2,1,2,1,2,1,2]):
        """
        Train conv nets: alpha and beta CNN.
        Currently, the two network does not share parameters.
        During each optimization process,
        we update parameters in two nets simultaneously.

        Parameters
        ----------
        training_iters: number of iterations during training
        learning_rate: learning rate
        batch_size: data size of mini-batch
        display_step: how many steps gap to show training result
        dropout: Dropout, probability to keep units
        strides: list with 8 elements
            1. conv layer 1 stride size for mean net
            2. max_pool layer 1 ksize and strides size for mean net
            3. conv layer 2 stride size for mean net
            4. max_pool layer 2 ksize and strides size for mean net
            5. conv layer 1 stride size for var net
            6. max_pool layer 1 ksize and strides size for var net
            7. conv layer 2 stride size for var net
            8. max_pool layer 2 ksize and strides size for var net
        """
        # Initializing network
        self.__construct_net__(strides=strides)
        self.__define_measure__()
        self.__optimization__(learning_rate=learning_rate)
        init = tf.global_variables_initializer()
        merged = tf.summary.merge_all()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Record train data
            # $ tensorboard --logdir=./summary_abnet
            train_writer = tf.summary.FileWriter('./tensorboard_abnet',
                                                  sess.graph)

            # Keep training until reach max iterations
            batch_x, batch_y = self.data.train.next_batch(batch_size)
            while step * batch_size <= training_iters:

                # Run optimization op (backprop)
                summary, _ = sess.run([merged, self.optimizer],
                                       feed_dict={self.x: batch_x,
                                                  self.y: batch_y,
                                                  self.keep_prob: dropout,
                                                  self.learning_rate: learning_rate})
                                                  #,self.sample_size: batch_y.shape[0]})
                train_writer.add_summary(summary, step)
                # display training intermediate result
                if step % display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([self.cost,self.accuracy],
                                          feed_dict={self.x: batch_x,
                                                     self.y: batch_y,
                                                     self.keep_prob: 1.})
                    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))

                step += 1

            print("Optimization finished")
            # Calculate test loss
            loss, acc = sess.run([self.cost,self.accuracy],
                                  feed_dict={self.x: self.data.test.images,
                                             self.y: self.data.test._labels,
                                             self.keep_prob: 1.})
            print("Test Loss= {:.6f}".format(loss) +", Test Accuracy= {:.5f}".format(acc))

if __name__ == '__main__':
    # read and munipulate data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    # Change labels 0~9 to {0,1}
    mnist_modify(mnist, {0:range(0,5), 1:range(5,10)}, one_hot=True)
    # Construct and train network
    abnn = alpha_beta_net(data = mnist)
    abnn.train_net(training_iters=40000, learning_rate=0.001,
                   batch_size=128, display_step=1)
