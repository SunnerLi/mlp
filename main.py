from tensorflow.examples.tutorials.mnist import input_data
from model import TinyGMM, MlpGMM
import tensorflow as tf
import numpy as np
import cv2
import sys
import os

if __name__ == '__main__':
    batch_size = 32
    noise_size = 100
    fc_size = 128
    image_size = 784
    noise_ph = tf.placeholder(tf.float32, [None, noise_size])
    image_ph = tf.placeholder(tf.float32, [None, image_size])
    loss_cheat_rate = tf.placeholder(tf.float32, [])
    net = MlpGMM(batch_size=batch_size, noise_size=noise_size, image_size=image_size, fc_size=fc_size)
    net.build(noise_ph, image_ph, loss_cheat_rate)
    result_path = './output'

    # Load data
    mnist = input_data.read_data_sets('./data')
    images = (mnist.train.images[:5000] - 127.5) / 127.5
    print(np.shape(images))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Train master discriminator
        for i in range(10):
            loss_list = []
            for j in range(len(images) // batch_size):
                feed_dict = {
                    noise_ph: np.random.random([batch_size, noise_size]),
                    image_ph: images[i * batch_size : i * batch_size + batch_size]
                }
                _d_loss, _ = sess.run([net.d_loss, net.master_optimize], feed_dict=feed_dict)
                loss_list.append(_d_loss)
            avg_loss = np.mean(np.asarray(loss_list))
            print('epoch: ', i, '\tmaster loss: ', avg_loss)
            if avg_loss < 1e-8:
                break

        # Train Generative Master Model
        _d_loss = 9.0
        _g_loss = 1.0
        for i in range(500):
            discriminator_loss_list = []
            generator_loss_list = []
            for j in range(len(images) // batch_size):
                if j % 100 == 0:
                    print('.', end='')
                    sys.stdout.flush()
                if _d_loss + _g_loss != 0:
                    if _d_loss < 10:
                        feed_dict = {
                            noise_ph: np.random.random([batch_size, noise_size]),
                            image_ph: images[i * batch_size : i * batch_size + batch_size],
                            loss_cheat_rate: _g_loss / (_d_loss + _g_loss)
                        }
                        _g_loss, _ = sess.run([net.g_loss, net.generator_optimize], feed_dict=feed_dict)
                    else:
                        feed_dict = {
                            noise_ph: np.random.random([batch_size, noise_size]),
                            image_ph: images[i * batch_size : i * batch_size + batch_size]
                        }
                        _g_loss, _ = sess.run([net.g_cheat_loss, net.generator_cheat_only_optimize], feed_dict=feed_dict)
                _d_loss, _ = sess.run([net.d_loss, net.discriminator_optimize], feed_dict=feed_dict)
                discriminator_loss_list.append(_d_loss)
                generator_loss_list.append(_g_loss)
            avg_g_loss = np.mean(np.asarray(generator_loss_list))
            avg_d_loss = np.mean(np.asarray(discriminator_loss_list))
            print('epoch: ', i, '\tgenerator loss: ', avg_g_loss, '\tmaster loss: ', avg_d_loss)

            # Visualization
            imgs = sess.run([net.g_tensor], feed_dict={
                noise_ph: np.random.random([batch_size, noise_size])
            })
            if not os.path.exists(result_path):
                os.mkdir(result_path)
            result = None
            imgs = imgs[0]
            for j in range(len(imgs) // 8):
                _res = None
                for k in range(8):
                    img = np.reshape(imgs[j*8+k], [28, 28])
                    if type(_res) == type(None):
                        _res = img
                    else:
                        _res = np.concatenate((_res, img), axis=1)
                if type(result) == type(None):
                    result = _res
                else:
                    result = np.concatenate((result, _res), axis=0)
            cv2.imwrite(result_path + '/' + str(i) + '.png', result)