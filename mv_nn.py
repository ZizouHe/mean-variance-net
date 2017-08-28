from mnist import MNIST
import numpy as np
import scipy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from base_model import conv_net, variable_summaries

class mean_var_net():
    def __init__(self, data, n_input = 784, n_classes = 2):
        self.data = data
        self.n_input = n_input
        self.n_classes = n_classes

    def __construct_net__(self, strides=[1,2,1,2,1,2,1,2]):
        with tf.variable_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=[None, self.n_input],name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes],name='y')

        with tf.variables_scope('hyperparameter'):
            self.keep_prob = tf.placeholder(tf.float32, name='dropout')
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        with tf.variables_scope('mean_net'):
            self.mean_net = conv_net('mean_net')
            self.mean_net.network(x=self.x, n_input=self.n_input, n_output=self.n_classes-1,
                                  dropout=self.keep_prob, strides=strides[:4],initializer="xavier")
            with tf.variables_scope('output'):
                self.mean_out = self.mean_net.out

        with tf.variables_scope('var_net'):
            self.var_net = conv_net('var_net')
            self.var_net.network(x=self.x, n_input=self.n_input, n_output=self.n_classes-1,
                                  dropout=self.keep_prob, strides=strides[4:],initializer="xavier")
            with tf.variables_scope('output'):
                self.var_out = 10*tf.sigmoid(self.mean_net.out)
                tf.histogram_summary("var_net/output/act_output", self.var_out)

    def __define_measure__(self):
        self.sample_size = tf.placeholder(tf.int32)
        with tf.variables_scope('accuracy'):


    def __optimization__(self, learning_rate=0.001):

    def train_net(self, training_iters = 20000, learning_rate = 0.001,
                   batch_size = 100, display_step = 1, dropout = 0.75,
                   strides = [1,1,2,2,1,1,2,2]):

        # Initializing the variables
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            batch_x, batch_y = self.data.train.next_batch(batch_size)
            # Keep training until reach max iterations
            while step * batch_size <= training_iters:

                # Run optimization op (backprop)
                #if step %2 == 1:
                mean,var = sess.run([self.mean_net,self.var_net], feed_dict={self.x: batch_x,
                                                self.y: batch_y,
                                                self.keep_prob: 1.,
                                                self.sample_size: batch_y.shape[0]})

                sess.run(self.apply_gradients, feed_dict={self.x: batch_x,
                                                          self.y: batch_y,
                                                          self.keep_prob: dropout,
                                                          self.sample_size: batch_y.shape[0]})

                # display training intermediate result

                if step % display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([self.cost,self.accuracy], feed_dict={self.x: batch_x,
                                                                                self.y: batch_y,
                                                                                self.sample_size: batch_y.shape[0],
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
    mnist.train.labels[mnist.train.labels <= 4] = 0
    mnist.train.labels[mnist.train.labels > 4] = 1
    mnist.test.labels[mnist.test.labels <= 4] = 0
    mnist.test.labels[mnist.test.labels > 4] = 1
    # modify labels for one_hot
    mnist.train._labels = np.eye(2)[mnist.train.labels]
    mnist.test._labels = np.eye(2)[mnist.test.labels]
    mvnn = mean_var_net(data=mnist)
    mvnn.train_net(training_iters=20000, learning_rate =0.001,batch_size=128, display_step=10)
