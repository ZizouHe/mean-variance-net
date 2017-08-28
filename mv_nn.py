from mnist import MNIST
import numpy as np
import scipy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from base_model import conv_net, variable_summaries

flags = tf.app.flags
FLAGS = flags.FLAGS

def cost_func(pred_mean, pred_var, y):
    scaler = tf.constant([-1,1], shape=[2,1], dtype=tf.float32)
    m_y = tf.matmul(y,scaler)

    dist = tf.contrib.distributions.Normal(loc=pred_mean, scale=pred_var)
    f = lambda x: 1/(1+tf.exp(-tf.multiply(x,m_y)))
    out = tf.contrib.bayesflow.monte_carlo.expectation(f=f,p=dist,n=1000)

    out = -tf.reduce_mean(tf.log(out + 1e-8))
    return out

def gradient_modify(pred_mean, pred_var, y, sample_size):
    scaler = tf.constant([1,-1], shape=[2,1], dtype=tf.float32)
    m_y = tf.matmul(y,scaler)
    dist = tf.contrib.distributions.Normal(loc=pred_mean, scale=pred_var)

    f1 = lambda x: 1/(1+tf.exp(-tf.multiply(x,m_y)))
    f2 = lambda x: tf.multiply(1/(1+tf.exp(-tf.multiply(x,m_y))),tf.div(x-pred_mean, pred_var**2))
    f3 = lambda x: tf.multiply(1/(1+tf.exp(-tf.multiply(x,m_y))),tf.div((pred_mean-x)**2, pred_var**3)-1/(pred_var**2))

    denominator = tf.contrib.bayesflow.monte_carlo.expectation(f=f1,p=dist,n=1000)
    numerator1 = tf.contrib.bayesflow.monte_carlo.expectation(f=f2,p=dist,n=1000)
    numerator2 = tf.contrib.bayesflow.monte_carlo.expectation(f=f3,p=dist,n=1000)
    sample_size = tf.cast(sample_size, tf.float32)
    out1 = tf.div(tf.div(numerator1,denominator), sample_size)
    out2 = tf.div(tf.div(numerator2,denominator), sample_size)

    return out1, out2

class mean_var_net():
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

        with tf.variable_scope('mean_net'):
            self.mean_net = conv_net('mean_net')
            self.mean_net.network(x=self.x, n_input=self.n_input,
                                  n_output=self.n_classes-1, dropout=self.keep_prob,
                                  strides=strides[:4],initializer="xavier")

        with tf.variable_scope('var_net'):
            self.var_net = conv_net('var_net')
            self.var_net.network(x=self.x, n_input=self.n_input,
                                 n_output=self.n_classes-1, dropout=self.keep_prob,
                                 strides=strides[4:],initializer="xavier")
            with tf.variable_scope('output'):
                self.var_net.out = 10*tf.sigmoid(self.var_net.out)
                tf.summary.histogram("var_net/output/act_output", self.var_net.out)

    def __define_measure__(self):
        self.sample_size = tf.placeholder(tf.int32)
        with tf.variable_scope('accuracy'):
            class_score = tf.concat([tf.zeros([self.sample_size, 1], tf.float32),
                                     self.mean_net.out],1)
            with tf.variable_scope('correct_prediction'):
                correct_pred = tf.equal(tf.argmax(class_score, 1),
                                        tf.argmax(self.y, 1))
            with tf.variable_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

            with tf.variable_scope('virtual_cost'):
                self.virtual_cost = tf.add(self.mean_net.out,self.var_net.out)
            with tf.variable_scope('true_cost'):
                self.cost = cost_func(self.mean_net.out,self.var_net.out,self.y)
                self.validation_cost = cost_func(self.mean_net.out,self.var_net.out,self.y)
            tf.summary.scalar('train_cost', self.cost)
            tf.summary.scalar('validation_cost', self.validation_cost)


    def __optimization__(self, learning_rate=0.001):
        with tf.variable_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            with tf.variable_scope("variables"):
                self.mean_variable = list(self.mean_net.variables.values())
                self.var_variable = list(self.var_net.variables.values())
            with tf.variable_scope("gradient"):
                self.grad_mod1,self.grad_mod2 = gradient_modify(self.mean_net.out, self.var_net.out,
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

    def train_net(self, training_iters = 20000, learning_rate = 0.001,
                   batch_size = 100, display_step = 1, dropout = 0.75,
                   strides = [1,2,1,2,1,2,1,2]):

        # Initializing the variables
        self.__construct_net__(strides=strides)
        self.__define_measure__()
        self.__optimization__(learning_rate=learning_rate)
        init = tf.global_variables_initializer()
        merged = tf.summary.merge_all()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            train_writer = tf.summary.FileWriter('../summary/train',
                                                  sess.graph)

            # Keep training until reach max iterations
            while step * batch_size <= training_iters:
                batch_x, batch_y = self.data.train.next_batch(batch_size)
                # Run optimization op (backprop)
                mean,var = sess.run([self.mean_net.out,self.var_net.out], feed_dict = {self.x: batch_x,
                                                                self.y: batch_y,
                                                                self.keep_prob: dropout})

                summary, _ = sess.run([merged, self.apply_gradients], feed_dict={self.x: batch_x,
                                                                                 self.y: batch_y,
                                                                                 self.keep_prob: dropout,
                                                                                 self.learning_rate: learning_rate,
                                                                                 self.sample_size: batch_y.shape[0]})

                sess.run(self.validation_cost, feed_dict={self.x: self.data.validation.images,
                                                          self.y: self.data.validation._labels,
                                                          self.sample_size: self.data.validation._labels.shape[0],
                                                          self.keep_prob: 1.})
                train_writer.add_summary(summary, step)
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
    mnist.validation.labels.setflags(write=1)
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
    mvnn = mean_var_net(data=mnist)
    mvnn.train_net(training_iters=20000, learning_rate =0.001,batch_size=128, display_step=2)
