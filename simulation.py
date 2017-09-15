import numpy as np
import tensorflow as tf
from math import sqrt
from data_generate import data_set,simulate_data

SAMPLE_SIZE = 2
NUM_CLASSES = 2
N_INPUT = 784
# Parameters, set seed
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
    learning_rate_virtual = 0.01
    learning_rate = 0.001
    training_iters = 50000
    batch_size = 128
    display_step = 10
    dropout = 0.75
    training_iters_virtual = 5000
    K = 5

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, N_INPUT])
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES+1])
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
    global_steps1 = tf.Variable(0,dtype=tf.int32,name='globel_steps1')
    global_steps2 = tf.Variable(0,dtype=tf.int32,name='globel_steps2')

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
        'wc1': tf.get_variable(
            name='wc1',
            shape=[5, 5, 1, 32],
            dtype=tf.float32,
            initializer=tf.contrib.layers.variance_scaling_initializer(uniform=False)),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.get_variable(
            name='wc2',
            shape=[5, 5, 32, 64],
            dtype=tf.float32,
            initializer=tf.contrib.layers.variance_scaling_initializer(uniform=False)),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.get_variable(
            name='wd1',
            shape=[k*64, 1024],
            dtype=tf.float32,
            initializer=tf.contrib.layers.variance_scaling_initializer(uniform=False)),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.get_variable(
            name='out',
            shape=[1024, NUM_CLASSES],
            dtype=tf.float32,
            initializer=tf.contrib.layers.variance_scaling_initializer(uniform=False))
    }

    biases = {
        'bc1': tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32)),
        'bc2': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32)),
        'bd1': tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32)),
        'out': tf.Variable(tf.constant(0.0, shape=[NUM_CLASSES], dtype=tf.float32))
    }

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)
    # virtual cost
    cost_virtual = tf.reduce_mean(tf.pow(tf.subtract(pred,2), 2))
    optimizer_virtual = tf.train.AdamOptimizer(
        learning_rate=learning_rate_virtual).minimize(
        cost_virtual)
    # pred = tf.nn.relu(pred)
    # true cost
    #pred = tf.nn.elu(pred) + 1
    #
    #pred = K*tf.nn.softmax(pred)

    def beta_likelihood(pred, y):

        deno = tf.multiply(
            tf.reduce_sum(pred,axis=1,keep_dims=True),
            tf.add(tf.reduce_sum(pred, axis=1,keep_dims=True),1))
        nomi1 = tf.reshape(tf.multiply(pred[:,0], pred[:,0]+1), shape=[-1,1])
        nomi2 = 2*tf.reshape(tf.multiply(pred[:,0], pred[:,1]), shape=[-1,1])
        nomi3 = tf.reshape(tf.multiply(pred[:,1], pred[:,1]+1), shape=[-1,1])
        likelihood = tf.div(
            tf.concat([nomi1, nomi2, nomi3],1),
            deno)
        return tf.reduce_sum(tf.multiply(likelihood, y),1)

    def bernoulli(pred, y):

        p = tf.nn.softmax(pred)
        nomi1 = tf.reshape(tf.pow(p[:,0], 2), shape=[-1,1])
        nomi2 = 2*tf.reshape(tf.multiply(p[:,0], p[:,1]), shape=[-1,1])
        nomi3 = tf.reshape(tf.pow(p[:,1], 2), shape=[-1,1])
        likelihood = tf.concat([nomi1, nomi2, nomi3], axis=1)

        return tf.reduce_sum(tf.multiply(likelihood,y),1), likelihood

    #likelihood = beta_likelihood(pred, y)
    likelihood,p = bernoulli(pred, y)

    # Define loss and optimizer
    cost = -tf.reduce_mean(tf.log(likelihood))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(
        cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # first virtual train
        """
        print("virtual train start...")
        while step*batch_size < training_iters_virtual:
            batch_x, batch_y = train_data.next_batch(batch_size)
            sess.run(optimizer_virtual, feed_dict={
                    x: batch_x,
                    y: batch_y,
                    keep_prob: 1.0})

            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss = sess.run(cost_virtual, feed_dict={
                    x: batch_x,
                    y: batch_y,
                    keep_prob: dropout})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss))
            step += 1
        print("virtual train finished.")

        step = 1
        """
        # Keep training until reach max iterations
        while step*batch_size < training_iters:
            batch_x, batch_y = train_data.next_batch(batch_size)
            sess.run(optimizer, feed_dict={
                    x: batch_x,
                    y: batch_y,
                    keep_prob: dropout})

            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss,prob,prd = sess.run([cost,p,pred], feed_dict={
                    x: batch_x,
                    y: batch_y,
                    keep_prob: dropout})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss))
                #print(prob)
                #print(prd)
                #break

            step += 1
        print("Optimization Finished!")

        # Calculate loss
        cst, prd = sess.run([cost,pred], feed_dict={x: test_data.images,
                                      y: test_data.labels,
                                      keep_prob: 1.})
        print("Testing loss: {:.6f}".format(loss))

        return prd

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def variable_summaries(var, name):
    with tf.variable_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        #with tf.variable_scope('stddev'):
        #    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        #tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

def initial_variable(name, shape, dtype, initializer, var=None):
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

class conv_net():
    def __init__(self, name):
        self.name = name
        self.variables = {}

    def nn_layer(self, input_dim, output_dim, layer_name, initializer="he"):
        var = 0.02/sqrt(np.prod(np.array(input_dim)))
        var = 5e-2
        with tf.variable_scope(layer_name, reuse=None):
            with tf.variable_scope("weight", reuse=None):
                self.variables[layer_name+'_w'] = initial_variable(name= 'weight',shape=input_dim+output_dim,
                                                                   dtype=tf.float32, initializer=initializer)


            with tf.variable_scope("bias", reuse=None):
                self.variables[layer_name+'_b'] = initial_variable(name= 'bias',shape=output_dim, dtype=tf.float32,
                                                                   initializer="constant")
        # Record variable summaries
        variable_summaries(self.variables[layer_name+'_w'], self.name+"/"+layer_name+'/weight')
        variable_summaries(self.variables[layer_name+'_b'], self.name+"/"+layer_name+'/bias')

    def __set_variable__(self, initializer, strides=[1,2,1,2]):
        """Set up network's variables"""

        full_size = int(sqrt(N_INPUT)*sqrt(N_INPUT)/(strides[1]**2)/(strides[3]**2))
        self.nn_layer([5,5,1], [32], "conv1", initializer)
        self.nn_layer([5,5,32], [64], "conv2", initializer)
        self.nn_layer([full_size*64], [1024], "fcon1", initializer)
        self.nn_layer([1024], [NUM_CLASSES-1], "output", initializer)

    def network(self, x, dropout, strides,initializer="he"):
        self.__set_variable__(strides=strides, initializer=initializer)

        x = tf.reshape(
            x,
            shape=[-1, int(sqrt(N_INPUT)), int(sqrt(N_INPUT)), 1])
        with tf.variable_scope('conv1', reuse=None):
            # Convolution Layer
            conv1 = conv2d(x, self.variables["conv1_w"], self.variables["conv1_b"], strides = strides[0])
            tf.summary.histogram(self.name+"/conv1/conv_output", conv1)
            # Max Pooling (down-sampling)
            conv1 = maxpool2d(conv1, k=strides[1])
            tf.summary.histogram(self.name+"/conv1/maxpool_output", conv1)
        with tf.variable_scope('conv2', reuse=None):
            # Convolution Layer
            conv2 = conv2d(conv1, self.variables["conv2_w"], self.variables["conv2_b"], strides = strides[2])
            tf.summary.histogram(self.name+"/conv2/conv_output", conv2)
            # Max Pooling (down-sampling)
            conv2 = maxpool2d(conv2, k=strides[3])
            tf.summary.histogram(self.name+"/conv2/maxpool_output", conv2)
        with tf.variable_scope('fcon1', reuse=None):
            # Fully connected layer
            # Reshape conv2 output to fit fully connected layer input
            fc1 = tf.reshape(conv2, [-1, self.variables["fcon1_w"].get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, self.variables["fcon1_w"]), self.variables["fcon1_b"])
            tf.summary.histogram(self.name+"/fcon1/fcon1_output", fc1)
            fc1 = tf.nn.relu(fc1)
            # Apply Dropout
            fc1 = tf.nn.dropout(fc1, dropout)
            tf.summary.histogram(self.name+"/fcon1/dropout_output", fc1)
        with tf.variable_scope('output', reuse=None):
            # Output
            self.out = tf.add(tf.matmul(fc1, self.variables["output_w"]), self.variables["output_b"])
            tf.summary.histogram(self.name+"/output/preact_output", self.out)

        return self.out

class conv_net2():
    def __init__(self, name):
        self.name = name
        self.variables = {}

    def nn_layer(self, input_dim, output_dim, layer_name, initializer="xavier"):
        var = 0.02/sqrt(np.prod(np.array(input_dim)))
        var = 5e-2
        with tf.variable_scope(layer_name, reuse=None):
            with tf.variable_scope("weight", reuse=None):
                self.variables[layer_name+'_w'] = initial_variable(name= 'weight',shape=input_dim+output_dim,
                                                                   dtype=tf.float32, initializer=initializer)


            with tf.variable_scope("bias", reuse=None):
                self.variables[layer_name+'_b'] = initial_variable(name= 'bias',shape=output_dim, dtype=tf.float32,
                                                                   initializer="constant")
        # Record variable summaries
        variable_summaries(self.variables[layer_name+'_w'], self.name+"/"+layer_name+'/weight')
        variable_summaries(self.variables[layer_name+'_b'], self.name+"/"+layer_name+'/bias')

    def __set_variable__(self, initializer):
        """Set up network's variables"""
        self.nn_layer([N_INPUT], [1024], "fcon1", initializer)
        self.nn_layer([1024], [1024], "fcon2", initializer)
        self.nn_layer([1024], [NUM_CLASSES-1], "output", initializer)

    def network(self, x, dropout, strides,initializer="xavier"):
        self.__set_variable__(initializer=initializer)
        with tf.variable_scope('fcon1', reuse=None):
            fc1 = tf.add(tf.matmul(x, self.variables["fcon1_w"]), self.variables["fcon1_b"])
            tf.summary.histogram(self.name+"/fcon1/fcon1_output", fc1)
            fc1 = tf.nn.elu(fc1)
            # Apply Dropout
            fc1 = tf.nn.dropout(fc1, dropout)
            tf.summary.histogram(self.name+"/fcon1/dropout_output", fc1)

        with tf.variable_scope('fcon2', reuse=None):
            fc2 = tf.add(tf.matmul(fc1, self.variables["fcon2_w"]), self.variables["fcon2_b"])
            tf.summary.histogram(self.name+"/fcon2/fcon2_output", fc2)
            fc2 = tf.nn.elu(fc2)
            # Apply Dropout
            fc2 = tf.nn.dropout(fc2, dropout)
            tf.summary.histogram(self.name+"/fcon2/dropout_output", fc2)
        with tf.variable_scope('output', reuse=None):
            # Output
            self.out = tf.add(tf.matmul(fc2, self.variables["output_w"]), self.variables["output_b"])
            tf.summary.histogram(self.name+"/output/preact_output", self.out)

        return self.out

class alpha_beta_net():
    """Alpha-Beta Network for Classification."""
    def __init__(self, data):
        self.data = data

    def __construct_net__(self, strides=[1,2,1,2]):
        with tf.variable_scope('input', reuse=None):
            self.x = tf.placeholder(tf.float32, shape=[None, N_INPUT],name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES+1],name='y')

        with tf.variable_scope('hyperparameter', reuse=None):
            self.keep_prob = tf.placeholder(tf.float32, name='dropout')
            tf.summary.scalar('dropout_keep_probability', self.keep_prob)
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            tf.summary.scalar('learning_rate', self.learning_rate)
            self.global_steps = tf.Variable(0,dtype=tf.int32,name='globel_steps')

        with tf.variable_scope('alpha_net', reuse=None):
            self.alpha_net = conv_net('alpha_net')
            # construct alpha network use truncated-normal initializer
            self.alpha_net.network(x=self.x, dropout=self.keep_prob,
                                   strides=strides[:4])
            #self.alpha_net.out = tf.nn.elu(self.alpha_net.out)+1

            with tf.variable_scope('output', reuse=None):
                #self.alpha_net.out = tf.sigmoid(self.alpha_net.out)
                variable_summaries(self.alpha_net.out, 'values')

        with tf.variable_scope('beta_net', reuse=None):
            self.beta_net = conv_net('beta_net')
            # construct alpha network use truncated-normal initializer
            self.beta_net.network(x=self.x,dropout=self.keep_prob,
                                   strides=strides[4:])
            self.beta_net.out = tf.nn.elu(self.beta_net.out)+1

            with tf.variable_scope('output', reuse=None):
                #self.beta_net.out = tf.sigmoid(self.beta_net.out)
                variable_summaries(self.beta_net.out, 'values')

    def __define_measure__(self):
        def beta_likelihood(pred1, pred2, y):
            pred1 = tf.nn.elu(pred1)+1
            pred2 = tf.nn.elu(pred2)+1
            deno = tf.multiply(
                tf.add(pred1,pred2),
                tf.add(tf.add(pred1,pred2),1))
            nomi1 = tf.reshape(tf.multiply(pred1, pred1+1), shape=[-1,1])
            nomi2 = 2*tf.reshape(tf.multiply(pred1, pred2), shape=[-1,1])
            nomi3 = tf.reshape(tf.multiply(pred2, pred2+1), shape=[-1,1])
            likelihood = tf.div(
                tf.concat([nomi1, nomi2, nomi3],1),
                deno)
            return likelihood, tf.reduce_sum(tf.multiply(likelihood, y),1)

        def bernoulli(pred1, pred2, y):
            p = tf.nn.softmax(tf.concat([pred1, pred2], axis=1))
            nomi1 = tf.reshape(tf.pow(p[:,0], 2), shape=[-1,1])
            nomi2 = 2*tf.reshape(tf.multiply(p[:,0], p[:,1]), shape=[-1,1])
            nomi3 = tf.reshape(tf.pow(p[:,1], 2), shape=[-1,1])
            likelihood = tf.concat([nomi1, nomi2, nomi3], axis=1)
            return likelihood, tf.reduce_sum(tf.multiply(likelihood,y),1)

        likelihood, true_value = beta_likelihood(self.alpha_net.out,self.beta_net.out,self.y)

        #likelihood, true_value = bernoulli(self.alpha_net.out,self.beta_net.out,self.y)

        with tf.variable_scope('accuracy', reuse=None):
            with tf.variable_scope('correct_prediction', reuse=None):
                correct_pred = tf.equal(tf.argmax(likelihood,1),
                                        tf.argmax(self.y,1))
            with tf.variable_scope('accuracy', reuse=None):
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.variable_scope('cost', reuse=None):
            self.cost = -tf.reduce_mean(tf.log(true_value))
            self.virtual_cost = tf.reduce_mean(
                tf.add(
                    tf.pow(tf.subtract(self.alpha_net.out,2),2),
                    tf.pow(tf.subtract(self.beta_net.out,2),2)))
            #self.validation_cost = cost_func(self.alpha_net.out,self.beta_net.out,self.y)
        tf.summary.scalar('train_cost', self.cost)
        #tf.summary.scalar('validation_cost', self.validation_cost)
        self.likelihood = likelihood

    def __optimization__(self, learning_rate=0.001):
        """Define optimization methods for networks."""
        with tf.variable_scope('optimization'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost, global_step=self.global_steps)
            self.virtual_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.virtual_cost)

    def train_net(self, training_iters = 40000, learning_rate = 0.001, batch_size = 128,
                  display_step = 10, dropout = 0.75, strides = [1,2,1,2,1,2,1,2]):
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
            print("virtual train start...")
            while step * 128 <= 30000:
                batch_x, batch_y = self.data.train.next_batch(128)
                # Run optimization op (backprop)
                sess.run(self.virtual_optimizer,
                                       feed_dict={self.x: batch_x,
                                                  self.y: batch_y,
                                                  self.keep_prob: dropout,
                                                  self.learning_rate: 0.003})
                #train_writer.add_summary(summary, step)
                # display training intermediate result
                if step % 10 == 0:
                    # Calculate batch loss and accuracy
                    loss = sess.run(self.virtual_cost,
                                          feed_dict={self.x: batch_x,
                                                     self.y: batch_y,
                                                     self.keep_prob: 1.})
                    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss))
                    a = input("continue?")
                    if a != "":
                        break

                step += 1
            step = 1
            print("virtual train finished...")
            # Record train data
            # $ tensorboard --logdir=./summary_abnet
            train_writer = tf.summary.FileWriter('./tensorboard_simunet',
                                                  sess.graph)

            # Keep training until reach max iterations
            #batch_x, batch_y = self.data.train.next_batch(batch_size)
            while step * batch_size <= training_iters:
                batch_x, batch_y = self.data.train.next_batch(batch_size)
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

            print("Optimization finished")
            # Calculate test loss
            loss, acc, lkh, alpha, beta = sess.run([self.cost,self.accuracy, self.likelihood,self.alpha_net.out, self.beta_net.out],
                                  feed_dict={self.x: self.data.test.images,
                                             self.y: self.data.test.labels,
                                             self.keep_prob: 1.})
            print("Test Loss= {:.6f}".format(loss) +", Test Accuracy= {:.5f}".format(acc))
            print(lkh)

            return alpha, beta

def main():
    #train_X, train_y = generate(50000)
    #test_X, test_y = generate(10000)

    #X,y = generate(70000)
    #data = simulate_data(X=X, y=y)

    #train_data = data_set(train_X, train_y)
    #test_data = data_set(test_X, test_y)
    #del train_X, train_y, test_X, test_y
    #pred = conv_train(train_data, test_data)
    data = simulate_data(
        path="./simulation_data",
        image_file="new_simulation_images.npz",
        label_file="new_simulation_labels.npz")
    abnet = alpha_beta_net(data)
    alpha, beta = abnet.train_net(
        training_iters=400000,
        learning_rate=0.01,
        batch_size=128,
        display_step=20,
        dropout=0.75)
    """
    logits = X.dot(W)+b
    p = np.exp(logits)/np.sum(np.exp(logits), axis=1,keepdims=True)
    p2 = pred/np.sum(pred,1,keepdims=True)
    """
    #return pred


if __name__ == '__main__':
    main()
