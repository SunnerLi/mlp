import tensorflow as tf

variable_list = []

def DenseLayer(input_tensor, unit, name='', reuse=None):
    if name == '':
        print('layer without name...')
    with tf.variable_scope(name, reuse=reuse):
        previous_size = input_tensor.get_shape()[1]
        weight = tf.get_variable('w', shape=[previous_size, unit], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        bias = tf.get_variable('b', shape=[unit], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        variable_list.append(weight)
        variable_list.append(bias)
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