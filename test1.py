from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import cv2

variable = []

def DenseLayer(input_tensor, unit, name='', reuse=None):
    if name == '':
        print('layer without name...')
    with tf.variable_scope(name, reuse=reuse):
        previous_size = input_tensor.get_shape()[1]
        weight = tf.get_variable('w', shape=[previous_size, unit], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        bias = tf.get_variable('b', shape=[unit], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        return tf.add(tf.matmul(input_tensor, weight), bias)

def reverse_DenseLayer(input_tensor, unit, name=''):
    with tf.variable_scope(name, reuse=True):
        previous_size = input_tensor.get_shape()[1]
        weight = tf.get_variable('w', shape=[unit, previous_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        bias = tf.get_variable('b', shape=[previous_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        transpose_weight = variable_inverse(weight, name=name)
        return tf.matmul(tf.subtract(input_tensor, bias), transpose_weight)

def variable_inverse(variable, name=''):
    if name == '':
        print('layer without name...')
    with tf.variable_scope(name, reuse=False):
        s, u, v = tf.svd(variable)
        return tf.matmul(tf.matmul(tf.transpose(v), tf.matrix_inverse(tf.diag(s))), tf.transpose(u))

def leaky_relu(input_tensor, alpha=0.2):
    return tf.maximum(alpha * input_tensor, input_tensor)

def reverse_leaky_relu(input_tensor, alpha=0.2):
    return tf.minimum(input_tensor / alpha, input_tensor)

def sigmoid(input_tensor):
    return 1 / (1 + tf.exp(-1 * input_tensor))

def reverse_sigmoid(input_tensor):
    return tf.matmul(sigmoid(input_tensor), 1 - sigmoid(input_tensor))

if __name__ == '__main__':
    noise_size = 100
    fc_size = 128
    image_size = 784
    noise_ph = tf.placeholder(tf.float32, [None, noise_size])
    image_ph = tf.placeholder(tf.float32, [None, image_size])

    # Construct generator
    g_fc1 = DenseLayer(noise_ph, unit=fc_size, name='g_fc1')
    g_lrelu1 = leaky_relu(g_fc1)
    g_tensor = DenseLayer(g_lrelu1, unit=image_size, name='g_fc2')

    # Construct true discriminator
    d_true_fc1 = DenseLayer(image_ph, unit=fc_size, name='d_fc1')
    d_true_lrelu1 = leaky_relu(d_true_fc1)
    d_true_fc2 = DenseLayer(d_true_lrelu1, unit=1, name='d_fc2')
    d_true_logits = sigmoid(d_true_fc2)

    # Construct fake discriminator
    d_fake_fc1 = DenseLayer(g_tensor, unit=fc_size, name='d_fc1', reuse=True)
    d_fake_lrelu1 = leaky_relu(d_fake_fc1)
    d_fake_fc2 = DenseLayer(d_fake_lrelu1, unit=1, name='d_fc2', reuse=True)
    d_fake_logits = sigmoid(d_fake_fc2)

    # Construct inverse discriminator  
    d_inv_sig = reverse_sigmoid(tf.ones_like(d_true_logits) - 0.5)
    d_inv_fc2 = reverse_DenseLayer(d_inv_sig, unit=fc_size, name='d_fc2')
    d_inv_lrelu = reverse_leaky_relu(d_inv_fc2)
    d_inv_logits = reverse_DenseLayer(d_inv_lrelu, unit=image_size, name='d_fc1')