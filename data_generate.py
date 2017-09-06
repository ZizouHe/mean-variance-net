#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 1 10:21:07 2017

@author: Zizou

generate simulation data
"""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from base_model import conv_net, variable_summaries, mnist_modify

class generation_net():
    """standard convnet for data generating"""
    def __init__(self, data, n_input = 784, n_classes = 2, num_examples=70000):
        """
        Initialize network

        Parameters
        ----------
        data: object from a data class,
              have method mini-batch, attributes train, test, validation
        n_input: data input size(e.g. img shape: 28*28 = 784)
        n_classes: total classes number(e.g. 0-9 digits; 0/1 labels)
        _index_in_epoch: index for next_train_batch method
        """
        self.data = data
        self.n_input = n_input
        self.n_classes = n_classes
        self._index_in_epoch = 0
        self._num_examples = num_examples

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

        with tf.variable_scope("generation_net", reuse=None):
            self.net = conv_net("generation_net")
            self.net.network(x=self.x, n_input=self.n_input,
                             n_output=self.n_classes, dropout=self.keep_prob,
                             strides=strides,initializer="he")
            with tf.variable_scope('output', reuse=None):
                variable_summaries(self.net.out, 'values')

    def __define_measure__(self):
        """
        Define cost and accuracy in mean var net.
        Record summaries to be shown in Tensorboard.

        Attributes
        ----------
        accuracy: classification accuracy
        cost: cost function on training set
        """
        with tf.variable_scope('accuracy', reuse=None):
            with tf.variable_scope('correct_prediction', reuse=None):
                correct_pred = tf.equal(tf.argmax(self.net.out,1),tf.argmax(self.y,1))
            with tf.variable_scope('accuracy', reuse=None):
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.variable_scope('cost', reuse=None):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.net.out,
                                                                               labels=self.y))
        tf.summary.scalar('train_cost', self.cost)

    def __optimization__(self, learning_rate=0.001):
        """Define optimization methods for networks."""
        with tf.variable_scope('optimization', reuse=None):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def __next_train_batch__(self, batch_size):
        """
        mini-batch method for training

        Parameters
        ----------
        batch_size: batch size

        Returns
        --------
        the next batch_size examples from this data set.
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # if finish in the epoch
        if self._index_in_epoch > self._num_examples:
            #shuffle the data
            perm = np.arrange(self._num_examples)
            np.random.shuffle(perm)
            self.data.train._images = self.data.train._images[perm]
            self.data.train._labels = self.data.train._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
            # check for batch_size scale
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self.data.train.images[start:end], self.data.train.labels[start:end]

    def train_net(self, training_iters = 20000, learning_rate = 0.001, batch_size = 100,
                  display_step = 1, dropout = 0.75, strides = [1,2,1,2]):
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
            print("Training start...")
            sess.run(init)
            step = 1
            # Record train data
            # $ tensorboard --logdir=./summary_gdnet
            train_writer = tf.summary.FileWriter('./tensorboard_gdnet',
                                                  sess.graph)

            # Keep training until reach max iterations
            while step * batch_size <= training_iters:
                batch_x, batch_y = self.__next_train_batch__(batch_size)
                # Run optimization op (backprop)
                summary, _ = sess.run([merged, self.optimizer],
                                       feed_dict={self.x: batch_x,
                                                  self.y: batch_y,
                                                  self.keep_prob: dropout,
                                                  self.learning_rate: learning_rate})
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

            print("Optimization finished...")
            # Calculate test loss
            self.pred, loss, acc = sess.run([self.net.out, self.cost,self.accuracy],
                                             feed_dict={self.x: self.data.train.images,
                                                        self.y: self.data.train.labels,
                                                        self.keep_prob: 1.})
            print("Train Loss= {:.6f}".format(loss) +", Train Accuracy= {:.5f}".format(acc))


def data_munipulate():
    """load and munipulate for generation"""
    # read data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    # Change labels 0~9 to {0,1}
    mnist_modify(mnist, {0:range(0,5), 1:range(5,10)}, one_hot=True)
    mnist.train.images.setflags(write=1)
    mnist.train.labels.setflags(write=1)
    mnist.train._images = np.concatenate((mnist.train.images,
                                          mnist.test.images,
                                          mnist.validation.images), axis=0)
    mnist.train._labels = np.concatenate((mnist.train.labels,
                                          mnist.test.labels,
                                          mnist.validation.labels), axis=0)
    print("Data munipulation finished...")
    return mnist

def data_generation(pred, X, y, sample_size=2):
    """
    Data generation function. Generate more than 1 labels for each input

    Parameters
    ----------
    pred: the un-softmax prediction, a.k.a. class score
          with size of [data_size, class_size]
    sample_size: how many to sample from 1 distribution

    Return
    ------
    1: input data, numpy matrix with size [data_size*sample_size, input_size]
    2: labels, numpy array/matrix with size [data_size*sample_size, class_size]
       each row is a label with one-hot representation
    """
    # drop wrong labeled data
    correct_pred = np.equal(np.argmax(pred,1),np.argmax(y,1))
    print("Total data size: {0}, correct labeled data size: {1}".format(correct_pred.shape[0],
                                                                        np.sum(correct_pred)))
    X = X[correct_pred, :]
    pred = pred[correct_pred, :]
    # softmax with soften operation
    dist = np.exp(pred/4) / np.sum(np.exp(pred/4),axis=1)[:, None]
    # generate random normal noise
    rand = np.random.normal(loc=0, scale=0.03*np.eye(dist.shape[1]-1),
                            size=(dist.shape[0],dist.shape[1]-1))
    # add noise to data
    dist = dist[:, :dist.shape[1]-1] + rand
    dist = np.concatenate((dist, 1 - np.sum(dist, axis=1)[:,None]), axis=1)
    dist = np.clip(dist, 0.02, 0.98)
    # define sample methods
    f = lambda x: np.random.choice(range(dist.shape[1]),size=sample_size, p=x)
    labels = np.apply_along_axis(f, 1, dist)
    print("Data Generation finished...")
    # return new input and labels
    return np.repeat(X, sample_size, axis=0), \
           np.eye(pred.shape[1])[np.reshape(labels, (np.prod(pred.shape),), order="F")]

class data_set():
    """data set"""
    def __init__(self, X, y):
        """
        initialized method, set write authority = False for labels

        Parameters
        ----------
        X: numpy matrix with size [data_size, input_size]
        y: numpy matrix(one-hot) or array with size [data_size, ?]
        """
        # check sample size
        if (X.shape[0] != y.shape[0]):
            raise ValueError("X and y should have same sample size!")
        self._images = X
        self._labels = y
        self._labels.setflags(write=0)
        self._index_in_epoch = 0
        self._num_examples = X.shape[0]

    def next_batch(self, batch_size):
        """
        mini-batch method for training

        Parameters
        ----------
        batch_size: batch size

        Returns
        --------
        the next batch_size examples from this data set.
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # if finish in the epoch
        if self._index_in_epoch > self._num_examples:
            #shuffle the data
            perm = np.arrange(self._num_examples)
            np.random.shuffle(perm)
            self.data.train._images = self.data.train._images[perm]
            self.data.train._labels = self.data.train._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
            # check for batch_size scale
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self.data.train.images[start:end], self.data.train.labels[start:end]

    @property
    def images(self):
        """read-only image attribute"""
        return self._images

    @property
    def labels(self):
        """read-only label attribute"""
        return self._labels


class simulate_data():
    """simulate data"""
    def __init__(self, X=None, y=None, path=".", image_file=None, label_file=None):
        """
        initialize method, divide data into 3 parts: train, test, validation

        Warning: Currently, we just shuffle the data, maybe it is necessary to
                 place data with same input in the same set.

        Parameters
        ----------
        X: input data, e.g. images
        y: input labels
        path: string, file path, if read from file
        image_file: string, image file name
        label_file: string, label file name

        Attributes
        ----------
        self.n_classes: class number
        self.train: training data
        self.test: test data
        self.validation: validation data
        """
        if (X is not None) and (y is not None):
            X = X
            y = y
        elif (image_file is not None) and (label_file is not None):
            X = np.load(path + "/" + image_file)
            y = np.load(path + "/" + label_file)
            print("Read data finished...")
        else:
            raise ValueError("enter valid data!")
        self.n_classes = y.shape[1]
        # randomly shuffle the data
        perm = np.arange(X.shape[0])
        np.random.shuffle(perm)
        train = perm[:int(len(perm)*0.79)]
        test = perm[int(len(perm)*0.79):int(len(perm)*0.93)]
        validation = perm[int(len(perm)*0.93):]
        self.train = data_set(X = X[train,:], y = y[train, :])
        self.test = data_set(X = X[test,:], y = y[test, :])
        self.validation = data_set(X = X[validation,:], y = y[validation, :])
        print("Split data finished...")

    def to_file(self, name, path="."):
        """save to file"""
        X = np.concatenate((self.train.images, self.test.images, self.validation.images), axis=0)
        y = np.concatenate((self.train.labels, self.test.labels, self.validation.labels), axis=0)
        np.save(path+"/"+name+"_images.npy", X)
        np.save(path+"/"+name+"_labels.npy", y)
        print("Data saved...")

def main():
    """main function"""
    # get data
    mnist = data_munipulate()
    # construct and train DNN
    gdnet = generation_net(data = mnist)
    gdnet.train_net(training_iters=50000,learning_rate=0.001,
                    batch_size=128, display_step=100, dropout=1)
    # sample new labels
    X,y = data_generation(pred=gdnet.pred.copy(), X=gdnet.data.train.images.copy(),
                           y=gdnet.data.train.labels.copy(), sample_size=2)
    data = simulate_data(X=X, y=y)
    data.to_file(name="simulation",path="./simulation_data")

    return gdnet

if __name__ == '__main__':
    main()
