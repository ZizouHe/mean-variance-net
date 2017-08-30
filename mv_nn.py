#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 19:27:03 2017

@author: Zizou

Construct mean-variance net
"""

from mnist import MNIST
import numpy as np
import scipy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from base_model import conv_net, variable_summaries


def cost_func(pred_mean, pred_var, y):
    """
    Custom cost function for mean-variance train.
    Our cost function is:
    $\sum_{i = 1}^n ln (E(\frac{1}{1 + e^{-yx}} | x \sim N[\mu; \sigma^2]))$
    Because cost function contains integral terms, we do not compute it directly;
    Instead, we use Monte-Carlo Simulation;
    We then use Monte-Carlo result to estimate the expectation = integral term.

    Parameters
    ----------
    pred_mean: a sample_size*(n_classes-1) matrix provides prob for all but one classes
    pred_var: a sample_size*(n_classes-1) matrix provides variance for all but one classes
    y: a sample_size*n_classes one-hot matrix provides labels
    note that pred_mean actually provides class score

    Returns
    -------
    a tensor
    """
    scaler = tf.constant([-1,1], shape=[2,1], dtype=tf.float32)
    # reshape y from n_size*n_class to n_size*(n_class-1) then calculate -y*x
    m_y = tf.matmul(y,scaler)

    dist = tf.contrib.distributions.Normal(loc=pred_mean, scale=pred_var)
    f = lambda x: 1/(1+tf.exp(-tf.multiply(x,m_y)))
    # Monte-Carlo expectation
    out = tf.contrib.bayesflow.monte_carlo.expectation(f=f,p=dist,n=1000)

    out = -tf.reduce_mean(tf.log(out + 1e-8))
    return out

def gradient_modify(pred_mean, pred_var, y, sample_size):
    """
    Gradient modify function.
    Calculate the last step gradient: gradient on cost function;
    Our gradient for mean net output is:
    $$
    \frac{1}{K}\sum_{i = 1}^K \frac{E\left\{\frac{1}{1 + \mathrm{e}^{-y_it}}
    \frac{t - \mu_(x_i)}{\sigma^2(x_i)}|\; t \sim N\left[\mu(x_i), \sigma^2(x_i)\right]\right\}}
    {E\left\{\frac{1}{1 + \mathrm{e}^{-y_it}} |\; t \sim N\left[\mu(x_i), \sigma^2(x_i)\right]\right\}}
    \frac{\partial \mu(x_i)}{\partial w_\mu}
    $$

    Our gradient for var net output is:
    $$
    \frac{1}{K}\sum_{i = 1}^K \frac{E\left\{\frac{1}{1 + \mathrm{e}^{-y_it}}
    \left[\frac{\left(t - \mu_(x_i)\right)^2}{\sigma^3(x_i)} \, - \frac{1}{\sigma(x_i)}\right]
    |\; t \sim N\left[\mu(x_i), \sigma^2(x_i)\right]\right\}}{E\left\{\frac{1}{1 + \mathrm{e}^{-y_it}}
    |\; t \sim N\left[\mu(x_i), \sigma^2(x_i)\right]\right\}} \frac{\partial \sigma(x_i)}{\partial w_\sigma}
    $$

    Parameters
    ----------
    pred_mean: a sample_size*(n_classes-1) matrix provides prob for all but one classes
    pred_var: a sample_size*(n_classes-1) matrix provides variance for all but one classes
    y: a sample_size*n_classes one-hot matrix provides labels
    note that pred_mean actually provides class score
    sample_size: sample size

    Returns
    -------
    a tuple of tensors, gradient of mean and variance net
    Note that each tensor has shape: [sample_size,1]
    """
    scaler = tf.constant([1,-1], shape=[2,1], dtype=tf.float32)
    # reshape y from n_size*n_class to n_size*(n_class-1) then calculate -y*x
    m_y = tf.matmul(y,scaler)
    dist = tf.contrib.distributions.Normal(loc=pred_mean, scale=pred_var)

    f1 = lambda x: 1/(1+tf.exp(-tf.multiply(x,m_y)))
    f2 = lambda x: tf.multiply(1/(1+tf.exp(-tf.multiply(x,m_y))),tf.div(x-pred_mean, pred_var**2))
    f3 = lambda x: tf.multiply(1/(1+tf.exp(-tf.multiply(x,m_y))),
                               tf.div((pred_mean-x)**2, pred_var**3)-1/(pred_var**2))

    denominator = tf.contrib.bayesflow.monte_carlo.expectation(f=f1,p=dist,n=1000)
    numerator1 = tf.contrib.bayesflow.monte_carlo.expectation(f=f2,p=dist,n=1000)
    numerator2 = tf.contrib.bayesflow.monte_carlo.expectation(f=f3,p=dist,n=1000)

    sample_size = tf.cast(sample_size, tf.float32)
    # gradient for mean net
    out1 = tf.div(tf.div(numerator1,denominator), sample_size)
    # gradient for var net
    out2 = tf.div(tf.div(numerator2,denominator), sample_size)

    return out1, out2

class mean_var_net():
    """Mean and Variance Network for Classification."""
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
        mean_net: mean network
        var_net: var network
        mean: mean network output
        var: var network output
        """
        with tf.variable_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=[None, self.n_input],name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes],name='y')

        with tf.variable_scope('hyperparameter'):
            self.keep_prob = tf.placeholder(tf.float32, name='dropout')
            tf.summary.scalar('dropout_keep_probability', self.keep_prob)
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            tf.summary.scalar('learning_rate', self.learning_rate)

        with tf.variable_scope('mean_net'):
            self.mean_net = conv_net('mean_net')
            # Use He's initialization
            self.mean_net.network(x=self.x, n_input=self.n_input,
                                  n_output=self.n_classes-1, dropout=self.keep_prob,
                                  strides=strides[:4], initializer="he")

            with tf.variable_scope('output'):
                # Add activation function on output, scale mean output to [-10,10]
                self.mean = 10*tf.tanh(self.mean_net.out)
                variable_summaries(self.mean, "values")

        with tf.variable_scope('var_net'):
            self.var_net = conv_net('var_net')
            # Use Xavier's initialization
            self.var_net.network(x=self.x, n_input=self.n_input,
                                 n_output=self.n_classes-1, dropout=self.keep_prob,
                                 strides=strides[4:], initializer="truncated")

            with tf.variable_scope('output'):
                # Add activation function on output, scale var output to [0.03,10.03]
                self.var = 10*tf.sigmoid(self.var_net.out)+0.03
                variable_summaries(self.var, 'values')

    def __define_measure__(self):
        """
        Define cost and accuracy in mean var net.
        Record summaries to be shown in Tensorboard.

        Attributes
        ----------
        accuracy: classification accuracy
        virtual_cost: fake cost used to compute gradient in previous steps
        cost: actual cost, computation needs Monte-Carlo simulation
        validation_cost: cost function value on validation set
        """
        self.sample_size = tf.placeholder(tf.int32)
        with tf.variable_scope('accuracy'):
            # compute class score
            class_score = tf.concat([tf.zeros([self.sample_size, 1], tf.float32),
                                     self.mean_net.out],1)
            with tf.variable_scope('correct_prediction'):
                correct_pred = tf.equal(tf.argmax(class_score, 1),
                                        tf.argmax(self.y, 1))
            with tf.variable_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

            with tf.variable_scope('virtual_cost'):
                self.virtual_cost = tf.add(self.mean,self.var)
            with tf.variable_scope('true_cost'):
                self.cost = cost_func(self.mean,self.var,self.y)
                self.validation_cost = cost_func(self.mean,self.var,self.y)
            tf.summary.scalar('train_cost', self.cost)
            tf.summary.scalar('validation_cost', self.validation_cost)


    def __optimization__(self, learning_rate=0.001):
        """
        Define optimization methods for networks.
        Use fake cost to calculate gradients on previous steps;
        Manually calculate the gradients on last step(actual cost);
        Combine the gradients and apply on Adam optimizer.

        Attributes
        ----------
        grad_mod1: mean net gradient on acutual cost function
        grad_mod2: var net gradient on acutual cost function
        apply_gradients: apply real gradients on network weights and biases
        """
        with tf.variable_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            with tf.variable_scope("variables"):
                self.mean_variable = list(self.mean_net.variables.values())
                self.var_variable = list(self.var_net.variables.values())
            with tf.variable_scope("gradient"):
                self.grad_mod1,self.grad_mod2 = gradient_modify(self.mean, self.var,
                                                                self.y, self.sample_size)

                gradient1 = optimizer.compute_gradients(loss=self.virtual_cost,
                                                        var_list=self.mean_variable,
                                                        grad_loss=self.grad_mod1)
                gradient2 = optimizer.compute_gradients(loss=self.virtual_cost,
                                                        var_list=self.var_variable,
                                                        grad_loss=self.grad_mod1)
                self.apply_gradients = optimizer.apply_gradients(gradient1+gradient2)

            variable_summaries(self.grad_mod1, "gradient/mean_gradient")
            variable_summaries(self.grad_mod2, "gradient/var_gradient")

    def train_net(self, training_iters = 20000, learning_rate = 0.001, batch_size = 100,
                  display_step = 1, dropout = 0.75, strides = [1,2,1,2,1,2,1,2]):
        """
        Train conv nets: mean and variance CNN.
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
            train_writer = tf.summary.FileWriter('./summary_data/train',
                                                  sess.graph)

            # Keep training until reach max iterations
            while step * batch_size <= training_iters:
                batch_x, batch_y = self.data.train.next_batch(batch_size)
                # Run optimization op (backprop)

                summary, _ = sess.run([merged, self.apply_gradients],
                                       feed_dict={self.x: batch_x,
                                                  self.y: batch_y,
                                                  self.keep_prob: dropout,
                                                  self.learning_rate: learning_rate,
                                                  self.sample_size: batch_y.shape[0]})


                train_writer.add_summary(summary, step)
                # display training intermediate result

                if step % display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([self.cost,self.accuracy],
                                          feed_dict={self.x: batch_x,
                                                     self.y: batch_y,
                                                     self.sample_size: batch_y.shape[0],
                                                     self.keep_prob: 1.})
                    # Calculate validation loss
                    sess.run(self.validation_cost,
                             feed_dict={self.x: self.data.validation.images,
                                        self.y: self.data.validation._labels,
                                        self.sample_size: self.data.validation._labels.shape[0],
                                        self.keep_prob: 1.})

                    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))


                step += 1

            print("Optimization Finished!")


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
    # Construct and train network
    mvnn = mean_var_net(data=mnist)
    mvnn.train_net(training_iters=40000, learning_rate =0.001,batch_size=128, display_step=10)
