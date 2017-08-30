#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:50:48 2017

@author: Zizou
"""

from base_model import conv_net, variable_summaries
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def cost_func(alpha, beta, y):
    return y

class alpha_beta_net():
    def __init__(self, data, n_input = 784, n_classes = 2):
        self.data = data
        self.n_input = n_input
        self.n_classes = n_classes

    def __construct_net__(self, strides=[1,2,1,2,1,2,1,2]):
        with tf.variable_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=[None, self.n_input],name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes],name='y')

        with tf.variable_scope('hyperparameter'):
            self.keep_prob = tf.placeholder(tf.float32, name='dropout')
            tf.summary.scalar('dropout_keep_probability', self.keep_prob)
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            tf.summary.scalar('learning_rate', self.learning_rate)

        with tf.variable_scope('alpha_net'):
            self.alpha_net = conv_net('alpha_net')
            self.alpha_net.network(x=self.x, n_input=self.n_input,
                                   n_output=self.n_classes-1, dropout=self.keep_prob,
                                   strides=strides[:4], initializer="xavier")

            with tf.variable_scope('output'):
                #self.alpha_net.out = tf.sigmoid(self.alpha_net.out)
                variable_summaries(self.alpha_net.out, 'values')

        with tf.variable_scope('beta_net'):
            self.alpha_net = conv_net('beta_net')
            self.alpha_net.network(x=self.x, n_input=self.n_input,
                                   n_output=self.n_classes-1, dropout=self.keep_prob,
                                   strides=strides[4:], initializer="xavier")

            with tf.variable_scope('output'):
                #self.beta_net.out = tf.sigmoid(self.beta_net.out)
                variable_summaries(self.beta_net.out, 'values')

    def __define_measure__(self):
        with tf.variable_scope('accuracy'):
            with tf.variable_scope('correct_prediction'):
                correct_pred = tf.equal(tf.argmax(tf.concat([self.alpha_net.out
                                                             self.beta_net.out],1),1),
                                        tf.argmax(self.y,1))
            with tf.variable_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.variable_scope('cost'):
            self.cost = cost_func(self.alpha_net.out,self.beta_net.out,self.y)
            self.validation_cost = cost_func(self.mean,self.var,self.y)
        tf.summary.scalar('train_cost', self.cost)
        tf.summary.scalar('validation_cost', self.validation_cost)

    def __optimization__(self, learning_rate=0.001):
        with tf.variable_scope('optimization'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def train_net(self, training_iters = 20000, learning_rate = 0.001, batch_size = 100,
                  display_step = 1, dropout = 0.75, strides = [1,2,1,2,1,2,1,2]):

        self.__construct_net__(strides=strides)
        self.__define_measure__()
        self.__optimization__(learning_rate=learning_rate)
        init = tf.global_variables_initializer()
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(init)
            step = 1

            train_writer = tf.summary.FileWriter('./summary_abnet',
                                                  sess.graph)

            while step * batch_size <= training_iters:
                batch_x, batch_y = self.data.train.next_batch(batch_size)

                summary, _ = sess.run([merged, self.optimizer],
                                       feed_dict={self.x: batch_x,
                                                  self.y: batch_y,
                                                  self.keep_prob: dropout,
                                                  self.learning_rate: learning_rate})
                                                  #,self.sample_size: batch_y.shape[0]})
                train_writer.add_summary(summary, step)

                if step % display_step == 0:
                    loss, acc = sess.run([self.cost,self.accuracy],
                                          feed_dict={self.x: batch_x,
                                                     self.y: batch_y,
                                                     self.keep_prob: 1.})
                    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))

                step += 1

            print("Optimization finished")
            loss, acc = sess.run([self.cost,self.accuracy],
                                  feed_dict={self.x: self.data.test.images,
                                             self.y: self.data.test._labels,
                                             self.keep_prob: 1.})
            print("Test Loss= {:.6f}".format(loss) +
                  ", Test Accuracy= {:.5f}".format(acc))

if __name__ == '__main__':
    # read and munipulate data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    # set attributes to write = True
    mnist.test.labels.setflags(write = 1)
    mnist.train.labels.setflags(write = 1)
    mnist.validation.labels.setflags(write=1)
    # Change labels 0~9 to {0,1}
    mnist.train.labels[mnist.train.labels <= 4] = 0
    mnist.train.labels[mnist.train.labels > 4] = 1
    mnist.test.labels[mnist.test.labels <= 4] = 0
    mnist.test.labels[mnist.test.labels > 4] = 1
    mnist.validation.labels[mnist.validation.labels <= 4] = 0
    mnist.validation.labels[mnist.validation.labels > 4] = 1
    # modify labels for one_hot
    mnist.train._labels = np.eye(2)[mnist.train.labels]
    mnist.test._labels = np.eye(2)[mnist.test.labels]
    mnist.validation._labels = np.eye(2)[mnist.validation.labels]
    abnn = alpha_beta_net(data = mnist)
    abnn.train_net(training_iters=20000, learning_rate=0.001, batch_size=128, display_step=5)
