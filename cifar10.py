#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:37:08 2017

@author: Zizou

"""
import tensorflow as tf
from cifar10_data import get_cifar
from data_generate import simulate_data
from datetime import datetime
import time
import re

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './simulation_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_integer('max_steps', 280,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('train_dir', './tensorboard_cifar10/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('eval_dir', './tensorboard_cifar10/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

# Global constants describing the CIFAR-10 data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 47400
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 8400
NUM_EXAMPLES_PER_EPOCH_FOR_VALIDATION = 4200
NUM_CLASSES = 10
IMAGE_SIZE = 24

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
#NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
#LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

# global dataset for training and testing
DATA = simulate_data(
    path="./simulation_data",
    image_file="cifar10_origin_images.npz",
    label_file="cifar10_origin_labels.npz")


def _activation_summary(x):
    """
    Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Parameters
    ----------
    x: Tensor

    Returns
    -------
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """
    Helper to create a Variable stored on CPU memory.

    Parameters
    ----------
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

    Returns
    -------
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Parameters
    ----------
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

    Returns
    -------
    Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def network(images, batch_size=FLAGS.batch_size):
    """
    Build the CIFAR-10 Conv_net model.

    Parameters
    ----------
    images: Images returned from distorted_inputs() or inputs().
    batch_size: batch_size

    Returns
    -------
    Logits.
    """
    # Instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1/192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear

def loss_func(logits, labels):
    """
    Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".

    Parameters
    ----------
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

    Returns
    -------
    Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def accuracy(logits, labels):
    """
    Compute accuracy

    Parameters
    ----------
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

    Returns
    -------
    Accuracy tensor of type float.
    """
    with tf.variable_scope('accuracy', reuse=None):
        with tf.variable_scope('correct_prediction', reuse=None):
            correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
        with tf.variable_scope('accuracy', reuse=None):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    return accuracy

def _add_loss_summaries(total_loss):
    """
    Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Parameters
    ----------
    total_loss: Total loss from loss().

    Returns
    -------
    loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def optimize(total_loss, global_step):
    """
    Train CIFAR-10 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Parameters
    ----------
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.

    Returns
    -------
    train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size

    '''
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    tf.summary.scalar('learning_rate', lr)
    '''

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute and apply gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,
        global_step)

    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def inputs(batch_size, eval_data=False):
    """
    Construct input for CIFAR evaluation

    Parameters
    ----------
    eval_data: bool, indicating if one should use the train or test data set.
    batch_size: Number of images per batch.

    Returns
    -------
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    # get next_batch data
    if not eval_data:
        X,y = DATA.train.next_batch(batch_size)
        print(DATA.train._index_in_epoch)
    else:
        X,y = DATA.test.next_batch(batch_size)

    def image_processing(X):
        """
        image munipulation

        Parameters
        ----------
        1D tensor or 1D numpy array

        Returns
        -------
        3D tensor, resized and reshaped image
        """
        # reshape inmage to a 3D tensor of [32, 32, 3]
        reshaped_image = tf.reshape(tf.cast(X, tf.float32), shape=[32,32,3])

        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # Image processing for evaluation.
        # Crop the central [height, width] of the image.
        resized_image = tf.image.resize_image_with_crop_or_pad(
            reshaped_image,
            height, width)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(resized_image)

        # Set the shapes of tensors.
        float_image.set_shape([height, width, 3])
        return float_image

    # reshape and resize
    X = tf.map_fn(image_processing, X, dtype=tf.float32)
    return X, y

def distorted_inputs(batch_size):
    """
    Construct input for CIFAR evaluation.
    Distorted inputs are generated for generalization.
    Note that this is different from inputs()
    we do not need to distorted test set

    Parameters
    ----------
    batch_size: Number of images per batch.

    Returns
    -------
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    if not eval_data:
        X,y = DATA.train.next_batch(batch_size)
    else:
        X,y = DATA.test.next_batch(batch_size)

    def image_distorting(X):
        """
        image munipulation

        Parameters
        ----------
        1D tensor or 1D numpy array

        Returns
        -------
        3D tensor, resized, reshaped and distorted image
        """
        # reshape inmage to a 3D tensor of [32, 32, 3]
        reshaped_image = tf.reshape(tf.cast(X, tf.float32), shape=[32,32,3])

        height = IMAGE_SIZE
        width = IMAGE_SIZE
        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                   max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                 lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)

        # Set the shapes of tensors.
        float_image.set_shape([height, width, 3])
        return float_image

    # reshape and resize
    X = tf.map_fn(image_distorting, X)
    return X, y

def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            images, labels = inputs(FLAGS.batch_size)
            """
            train_images, train_labels = inputs(
                batch_size=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

            test_images, test_labels = inputs(
                batch_size=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,
                eval_data=True)
            """

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = network(images)

        # Calculate loss.
        loss = loss_func(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = optimize(loss, global_step)
        start = time.time()
        print("\nStart training...")

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
            tf.train.NanTensorHook(loss),
               _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement)) as mon_sess:

            while not mon_sess.should_stop():
                mon_sess.run(train_op)

        end = time.time()
        print("Training process finished, total time %.2fs" % (end - start))

def _eval(train,batch_size):
    with tf.Graph().as_default() as g:
        with tf.device('/cpu:0'):
            if train:
                op = "train"
                images, labels = inputs(
                    batch_size=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
            else:
                op = "test"
                images, labels = inputs(
                    batch_size=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,
                    eval_data=True)

        logits = network(images, batch_size=batch_size)
        loss = loss_func(logits, labels)
        # Calculate predictions.
        top_k_op = accuracy(logits, labels)
        with tf.Session() as sess:
            print("%s loss: %.4f, %s accuracy: %.2f" % (op, loss.eval(), op, top_k_op.eval()))


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """
    Run Eval once.

    Parameters
    ----------
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))

        num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * FLAGS.batch_size
        step = 0
        while step < num_iter and not coord.should_stop():
            predictions = sess.run([top_k_op])
            true_count += np.sum(predictions)
            step += 1

        # Compute precision @ 1.
        precision = true_count / total_sample_count
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op))
        summary.value.add(tag='Precision @ 1', simple_value=precision)
        summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

def evaluate(eval_data, batch_size):
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        images, labels = inputs(
            eval_data=eval_data,
            batch_size=batch_size)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = network(images, batch_size)

        # Calculate predictions.
        top_k_op = accuracy(logits, labels)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
        time.sleep(FLAGS.eval_interval_secs)

if __name__ == '__main__':
    train()
