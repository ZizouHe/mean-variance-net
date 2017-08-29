#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Fri 14 10:58:27 2017

@author: Zizou
"""
# Construct MNIST CNN Network Using Dropout trick

import numpy as np
import tensorflow as tf
from MNDATA import mndata
import sklearn.metrics as metrics

# Import MNIST data
data = mndata()
# Relabeling, use 2 class instead: 0~4 assigned 0; otherwise 1
data.relabel(2, [range(5), range(5,10)])
data.one_hot()

# Parameters Setting
learning_rate = 0.001
training_iters = 200000
batch_size = 100
display_step = 100
N_model = 50

# Network Parameters
n_input = 784 # MNIST data input (image shape: 28*28)
n_classes = 2 # Instead of 10
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, 2]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([2]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
# Record the probability
prob_pred = tf.nn.softmax(pred)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# accu:
accu = []
prob = []

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    print("Training start...")
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = data.next_batch(batch_size = batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    for i in range(N_model):
        # Calculate accuracy for 10000 mnist test images, and record it
        accu.append(sess.run(accuracy, feed_dict={x: data.X_test,
                                                  y: data.labels_test,
                                                  keep_prob: dropout}))
        print("Testing Accuracy on {0}th dropout:".format(i+1), accu[-1])
        # Record probability calculated
        prob.append(sess.run(prob_pred, feed_dict = {x: data.X_test,
                                                     keep_prob: dropout}))

labels_pred = []
for i in range(N_model):
    if i == 0:
        labels_pred = prob[i]
    else:
        labels_pred += prob[i]

labels_pred /= N_model
print("Test accuracy, ensemble model: {0}".format(metrics.accuracy_score
    (np.argmax(data.labels_test, 1), np.argmax(labels_pred, 1))))

print("Test accuracy, each individual model:", accu)

# Construct Prediction Inverval for predicting P(y = 0 | x)
label0_pred = labels_pred[:, 0]
# Calculate sigma estimation
sigma_pred = []
for i in range(N_model):
    if i == 0:
        sigma_pred = pow((label0_pred - prob[i][:, 0]), 2)
    else:
        sigma_pred += pow((label0_pred - prob[i][:, 0]), 2)
sigma_pred /= (N_model - 1)
# Construct PI
PI = np.zeros([len(label0_pred), 2])
PI[:, 0] = label0_pred - sigma_pred
PI[:, 1] = label0_pred + sigma_pred

# Measurement on PI
crit_value = 0.05
print("Test accuracy on overall model: {:.5f}".format(metrics.accuracy_score
    (np.argmax(data.labels_test, 1), np.argmax(labels_pred, 1))))
print("Test accuracy on wide PI: {:.5f}".format(metrics.accuracy_score
    (np.argmax(labels_pred, 1)[sigma_pred > crit_value],
        np.argmax(data.labels_test, 1)[sigma_pred > crit_value])))

print("Test accuracy on narrow PI: {:.5f}".format(metrics.accuracy_score
    (np.argmax(labels_pred, 1)[sigma_pred <= crit_value],
        np.argmax(data.labels_test, 1)[sigma_pred <= crit_value])))
