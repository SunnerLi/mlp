from layer import *
import tensorflow as tf

class TinyGMM(object):
    def __init__(self, batch_size=32, noise_size=100, image_size=784, fc_size=128):
        self.batch_size = batch_size
        self.noise_size = noise_size
        self.image_size = image_size
        self.fc_size = fc_size

    def build(self, noise_ph, image_ph, loss_cheat_rate):
        # Construct network
        self.getGenerator(noise_ph, name='generator')
        self.getDiscriminator(noise_ph, image_ph, name='discriminator')

        # Define loss function
        d_true_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_true_logits, labels=tf.ones_like(self.d_true_logits)))
        d_false_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logits, labels=tf.zeros_like(self.d_fake_logits)))
        self.d_loss = d_true_loss + d_false_loss
        g_cheat_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logits, labels=tf.ones_like(self.d_fake_logits)))	
        g_teach_loss = tf.reduce_mean(tf.square(self.d_inv_label - self.g_tensor))
        # g_teach_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=g_tensor, labels=d_inv_label))
        self.g_loss = loss_cheat_rate * g_cheat_loss + (1 - loss_cheat_rate) * g_teach_loss

        # Define optimizer
        generator_vars = [tensor for tensor in tf.trainable_variables() if 'generator' in tensor.name]
        discriminator_vars = [tensor for tensor in tf.trainable_variables() if 'discriminator' in tensor.name]
        print(generator_vars)
        print(discriminator_vars)
        self.master_optimize = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.d_loss, var_list=discriminator_vars)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.discriminator_optimize = tf.train.AdamOptimizer().minimize(self.d_loss, var_list=discriminator_vars)
            self.generator_optimize = tf.train.AdamOptimizer().minimize(self.g_loss, var_list=generator_vars)

    def getGenerator(self, noise_ph, name=None):
        if name == None:
            raise Exception('You should assign the name of generator')
        with tf.variable_scope('generator'):
            self.g_fc1 = DenseLayer(noise_ph, unit=self.fc_size, name='g_fc1')
            self.g_lrelu1 = leaky_relu(self.g_fc1)
            self.g_tensor = DenseLayer(self.g_lrelu1, unit=self.image_size, name='g_fc2')

    def getDiscriminator(self, noise_ph, image_ph, name=None):
        if name == None:
            raise Exception('You should assign the name of discriminator')
        with tf.variable_scope(name):
            # Construct true discriminator
            self.d_true_fc1 = DenseLayer(image_ph, unit=self.fc_size, name='d_fc1')
            self.d_true_lrelu1 = leaky_relu(self.d_true_fc1)
            self.d_true_fc2 = DenseLayer(self.d_true_lrelu1, unit=1, name='d_fc2')
            # self.d_true_logits = sigmoid(d_true_fc2)
            self.d_true_logits = self.d_true_fc2

            # Construct fake discriminator
            self.d_fake_fc1 = DenseLayer(self.g_tensor, unit=self.fc_size, name='d_fc1', reuse=True)
            self.d_fake_lrelu1 = leaky_relu(self.d_fake_fc1)
            self.d_fake_fc2 = DenseLayer(self.d_fake_lrelu1, unit=1, name='d_fc2', reuse=True)
            # self.d_fake_logits = sigmoid(d_fake_fc2)
            self.d_fake_logits = self.d_fake_fc2

            # Construct inverse discriminator  
            # d_inv_sig = tf.ones_like(d_true_logits)
            self.d_inv_sig = tf.truncated_normal(tf.shape(self.d_true_logits), mean=0.9, stddev=0.05)
            self.d_inv_fc2 = reverse_DenseLayer(self.d_inv_sig, unit=self.fc_size, name='d_fc2')
            self.d_inv_lrelu = reverse_leaky_relu(self.d_inv_fc2)
            self.d_inv_label = reverse_DenseLayer(self.d_inv_lrelu, unit=self.image_size, name='d_fc1')


class MlpGMM(object):
    def __init__(self, batch_size=32, noise_size=100, image_size=784, fc_size=32):
        self.batch_size = batch_size
        self.noise_size = noise_size
        self.image_size = image_size
        self.fc_size = fc_size

    def build(self, noise_ph, image_ph, loss_cheat_rate):
        # Construct network
        self.getGenerator(noise_ph, name='generator')
        self.getDiscriminator(noise_ph, image_ph, name='discriminator')

        # Define loss function
        d_true_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_true_logits, labels=tf.ones_like(self.d_true_logits)))
        d_false_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logits, labels=tf.zeros_like(self.d_fake_logits)))
        self.d_loss = d_true_loss + d_false_loss
        self.g_cheat_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logits, labels=tf.ones_like(self.d_fake_logits)))	
        g_teach_loss = tf.reduce_mean(tf.square(self.d_inv_label - self.g_tensor))
        # g_teach_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=g_tensor, labels=d_inv_label))
        self.g_loss = loss_cheat_rate * self.g_cheat_loss + (1 - loss_cheat_rate) * g_teach_loss

        # Define optimizer
        generator_vars = [tensor for tensor in tf.trainable_variables() if 'generator' in tensor.name]
        discriminator_vars = [tensor for tensor in tf.trainable_variables() if 'discriminator' in tensor.name]
        print(generator_vars)
        print(discriminator_vars)
        self.master_optimize = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.d_loss, var_list=discriminator_vars)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.discriminator_optimize = tf.train.AdamOptimizer().minimize(self.d_loss, var_list=discriminator_vars)
            self.generator_optimize = tf.train.AdamOptimizer().minimize(self.g_loss, var_list=generator_vars)
            self.generator_cheat_only_optimize = tf.train.AdamOptimizer().minimize(self.g_cheat_loss, var_list=generator_vars)

    def getGenerator(self, noise_ph, name=None):
        if name == None:
            raise Exception('You should assign the name of generator')
        with tf.variable_scope('generator'):
            self.g_fc1 = leaky_relu(DenseLayer(noise_ph, unit=self.fc_size, name='g_fc1'))
            self.g_fc2 = leaky_relu(DenseLayer(self.g_fc1, unit=self.fc_size, name='g_fc2'))
            self.g_fc3 = leaky_relu(DenseLayer(self.g_fc2, unit=self.fc_size, name='g_fc3'))
            self.g_tensor = DenseLayer(self.g_fc3, unit=self.image_size, name='g_fc4')

    def getDiscriminator(self, noise_ph, image_ph, name=None):
        if name == None:
            raise Exception('You should assign the name of discriminator')
        with tf.variable_scope(name):
            # Construct true discriminator
            self.d_true_fc1 =leaky_relu(DenseLayer(image_ph, unit=self.fc_size, name='d_fc1'))
            self.d_true_fc2 =leaky_relu(DenseLayer(self.d_true_fc1, unit=self.fc_size, name='d_fc2'))
            self.d_true_fc3 =leaky_relu(DenseLayer(self.d_true_fc2, unit=self.fc_size, name='d_fc3'))
            self.d_true_logits = DenseLayer(self.d_true_fc3, unit=1, name='d_fc4')

            # Construct fake discriminator
            self.d_fake_fc1 = leaky_relu(DenseLayer(self.g_tensor, unit=self.fc_size, name='d_fc1', reuse=True))
            self.d_fake_fc2 = leaky_relu(DenseLayer(self.d_fake_fc1, unit=self.fc_size, name='d_fc2', reuse=True))
            self.d_fake_fc3 = leaky_relu(DenseLayer(self.d_fake_fc2, unit=self.fc_size, name='d_fc3', reuse=True))
            self.d_fake_logits = DenseLayer(self.d_fake_fc3, unit=1, name='d_fc4', reuse=True)

            # Construct inverse discriminator  
            self.d_inv_sig = tf.truncated_normal(tf.shape(self.d_true_logits), mean=0.9, stddev=0.05)
            self.d_inv_fc4 = reverse_leaky_relu(reverse_DenseLayer(self.d_inv_sig, unit=self.fc_size, name='d_fc4'))
            self.d_inv_fc3 = reverse_leaky_relu(reverse_DenseLayer(self.d_inv_fc4, unit=self.fc_size, name='d_fc3'))
            self.d_inv_fc2 = reverse_leaky_relu(reverse_DenseLayer(self.d_inv_fc3, unit=self.fc_size, name='d_fc2'))
            self.d_inv_label = reverse_DenseLayer(self.d_inv_fc2, unit=self.image_size, name='d_fc1')