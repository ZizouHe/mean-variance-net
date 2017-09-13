import numpy as np
import tensorflow as tf
from math import sqrt
from data_generate import data_set

SAMPLE_SIZE = 2
NUM_CLASSES = 2
N_INPUT = 400
# Parameters, setseed
np.random.seed(seed=1)
W = np.random.normal(0,0.1,size=(N_INPUT, NUM_CLASSES))
b = np.random.normal(0,0.1,size=(NUM_CLASSES,))
# reset seed
np.random.seed(seed=None)

def generate(data_size=10000):
    # data
    X = np.random.uniform(0,1, size=(data_size,N_INPUT))
    #noise
    Z = np.random.normal(0,2, size=(data_size, NUM_CLASSES))
    # calculate p
    logits = X.dot(W)+b+Z
    p = np.exp(logits)/np.sum(np.exp(logits), axis=1,keepdims=True)

    # sample
    f = lambda x: np.random.choice(range(p.shape[1]),size=SAMPLE_SIZE, p=x)
    labels = np.sum(np.apply_along_axis(f, 1, p), axis=1)
    labels = np.eye(SAMPLE_SIZE+1)[labels]

    return X, labels

def conv_train(train_data, test_data):
    # Parameters
    learning_rate = 0.001
    training_iters = 40000
    batch_size = 128
    display_step = 10
    dropout = 0.75

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, N_INPUT])
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES+1])
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
        x = tf.reshape(
            x,
            shape=[-1, int(sqrt(N_INPUT)), int(sqrt(N_INPUT)), 1])

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

    k = int(N_INPUT/16)
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([k*64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, NUM_CLASSES]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([NUM_CLASSES]))
    }

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step*batch_size < training_iters:
            batch_x, batch_y = train_data.next_batch(batch_size)
            sess.run(optimizer, feed_dict={
                    x: batch_x,
                    y: batch_y,
                    keep_prob: dropout})

            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss = sess.run(cost, feed_dict={
                    x: batch_x,
                    y: batch_y,
                    keep_prob: dropout})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss))
            step += 1
        print("Optimization Finished!")

        # Calculate loss
        print("Testing Accuracy:", \
            sess.run(cost, feed_dict={x: test_data.images,
                                          y: test_data.labels,
                                          keep_prob: 1.}))

def main():
    train_X, train_y = generate(10000)
    test_X, test_y = generate(2000)
    train_data = data_set(train_X, train_y)
    test_data = data_set(test_X, test_y)
    del train_X, train_y, test_X, test_y
    conv_train(train_data, test_data)


if __name__ == '__main__':
    main()
