import tensorflow as tf
import numpy as np

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout, strides):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], strides = strides[0])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=strides[2])

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], strides = strides[1])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=strides[3])

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class score/variance
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def cost_func(pred_mean, pred_var, y, sample_size, mtcl_size=500):

    return out

def gradient_modify(pred_mean, pred_var, y, sample_size, mtcl_size=500):
    out1 = tf.div(numerator1,denominator)
    out2 = tf.div(numerator2,denominator)
    return out1,out2


def optimizer(cost_func, learning_rate):

    gradient = tf.train.AdamOptimizer(learning_rate=learning_rate).compute_gradients(loss=cost_func)
    return gradient
