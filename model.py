import tensorflow as tf
from tensorflow.contrib.layers import flatten


def conv2d(input, shape):
    conv = tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=0.1))
    bias = tf.Variable(tf.zeros(shape[-1]))
    conv = tf.nn.conv2d(input, conv, strides=[1, 1, 1, 1], padding='VALID') + bias
    return conv


def linear(input, shape):
    fc = tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=0.1))
    bias = tf.Variable(tf.zeros(shape[-1]))
    fc = tf.matmul(input, fc) + bias
    return fc


def LeNet(images):
    # images = tf.reshape(images, shape=[-1, 28, 28, 1])

    # Conv 1
    # conv1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=0, stddev=0.1))
    # conv1_b = tf.Variable(tf.zeros(6))
    # conv1 = tf.nn.conv2d(images, conv1, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    # conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.relu(conv2d(input=images, shape=(5, 5, 1, 6)))

    # Pool 1
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Conv 2
    # conv2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=0, stddev=0.1))
    # conv2_b = tf.Variable(tf.zeros(16))
    # conv2 = tf.nn.conv2d(conv1, conv2, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    # conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.relu(conv2d(input=conv1, shape=(5, 5, 6, 16)))

    # Pool 2
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Flatten
    fc0 = flatten(conv2)

    # Full 1
    # fc1 = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=0, stddev=0.1))
    # fc1_b = tf.Variable(tf.zeros(120))
    # fc1 = tf.matmul(fc0, fc1) + fc1_b
    # fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.relu(linear(input=fc0, shape=(400, 120)))

    # Full 2
    # fc2 = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=0, stddev=0.1))
    # fc2_b = tf.Variable(tf.zeros(84))
    # fc2 = tf.matmul(fc1, fc2) + fc2_b
    # fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.relu(linear(input=fc1, shape=(120, 84)))

    # Out
    # fc3 = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=0, stddev=0.1))
    # fc3_b = tf.Variable(tf.zeros(10))
    # output = tf.matmul(fc2, fc3) + fc3_b
    output = linear(input=fc2, shape=(84, 10))

    return output
