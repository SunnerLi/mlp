from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import cv2

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

if __name__ == '__main__':
    batch_size = 1
    noise_size = 100
    fc_size = 128
    image_size = 784
    noise_ph = tf.placeholder(tf.float32, [None, noise_size])
    image_ph = tf.placeholder(tf.float32, [None, image_size])
    loss_cheat_rate = tf.placeholder(tf.float32, [])

    # Construct generator
    with tf.variable_scope('generator'):
        g_fc1 = DenseLayer(noise_ph, unit=fc_size, name='g_fc1')
        g_lrelu1 = leaky_relu(g_fc1)
        g_tensor = DenseLayer(g_lrelu1, unit=image_size, name='g_fc2')

    with tf.variable_scope('discriminator'):
        # Construct true discriminator
        d_true_fc1 = DenseLayer(image_ph, unit=fc_size, name='d_fc1')
        d_true_lrelu1 = leaky_relu(d_true_fc1)
        d_true_fc2 = DenseLayer(d_true_lrelu1, unit=1, name='d_fc2')
        # d_true_logits = sigmoid(d_true_fc2)
        d_true_logits = d_true_fc2

        # Construct fake discriminator
        d_fake_fc1 = DenseLayer(g_tensor, unit=fc_size, name='d_fc1', reuse=True)
        d_fake_lrelu1 = leaky_relu(d_fake_fc1)
        d_fake_fc2 = DenseLayer(d_fake_lrelu1, unit=1, name='d_fc2', reuse=True)
        # d_fake_logits = sigmoid(d_fake_fc2)
        d_fake_logits = d_fake_fc2

        # Construct inverse discriminator  
        # d_inv_sig = tf.ones_like(d_true_logits)
        d_inv_sig = tf.truncated_normal(tf.shape(d_true_logits), mean=0.9, stddev=0.05)
        d_inv_fc2 = reverse_DenseLayer(d_inv_sig, unit=fc_size, name='d_fc2')
        d_inv_lrelu = reverse_leaky_relu(d_inv_fc2)
        d_inv_label = reverse_DenseLayer(d_inv_lrelu, unit=image_size, name='d_fc1')

    # Define loss function
    d_true_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_true_logits, labels=tf.ones_like(d_true_logits)))
    d_false_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits)))
    d_loss = d_true_loss + d_false_loss
    g_cheat_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_logits)))	
    g_teach_loss = tf.reduce_mean(tf.square(d_inv_label - g_tensor))
    # g_teach_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=g_tensor, labels=d_inv_label))
    g_loss = loss_cheat_rate * g_cheat_loss + (1 - loss_cheat_rate) * g_teach_loss

    # Define optimizer
    generator_vars = [tensor for tensor in tf.trainable_variables() if 'generator' in tensor.name]
    discriminator_vars = [tensor for tensor in tf.trainable_variables() if 'discriminator' in tensor.name]
    print(generator_vars)
    print(discriminator_vars)
    master_optimize = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(d_loss, var_list=discriminator_vars)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        discriminator_optimize = tf.train.AdamOptimizer().minimize(d_loss, var_list=discriminator_vars)
        generator_optimize = tf.train.AdamOptimizer().minimize(g_loss, var_list=generator_vars)

    # Load data
    mnist = input_data.read_data_sets('./data')
    images = (mnist.train.images[:5000] - 127.5) / 127.5
    print(np.shape(images))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Train master discriminator
        for i in range(5):
            loss_list = []
            for j in range(len(images) // batch_size):
                feed_dict = {
                    noise_ph: np.random.random([batch_size, noise_size]),
                    image_ph: images[i * batch_size : i * batch_size + batch_size]
                }
                _d_loss, _ = sess.run([d_loss, master_optimize], feed_dict=feed_dict)
                loss_list.append(_d_loss)
            avg_loss = np.mean(np.asarray(loss_list))
            print('epoch: ', i, '\tmaster loss: ', avg_loss)
            if avg_loss == 0:
                break

        # Train Generative Master Model
        _d_loss = 9.0
        _g_loss = 1.0
        for i in range(5):
            discriminator_loss_list = []
            generator_loss_list = []
            for j in range(len(images) // batch_size):
                feed_dict = {
                    noise_ph: np.random.random([batch_size, noise_size]),
                    image_ph: images[i * batch_size : i * batch_size + batch_size],
                    loss_cheat_rate: _g_loss / (_d_loss + _g_loss)
                }
                _g_loss, _ = sess.run([g_loss, generator_optimize], feed_dict=feed_dict)
                _d_loss, _ = sess.run([d_loss, discriminator_optimize], feed_dict=feed_dict)
                discriminator_loss_list.append(_d_loss)
                generator_loss_list.append(_g_loss)
            avg_g_loss = np.mean(np.asarray(generator_loss_list))
            avg_d_loss = np.mean(np.asarray(discriminator_loss_list))
            print('epoch: ', i, '\tgenerator loss: ', avg_g_loss, '\tmaster loss: ', avg_d_loss)