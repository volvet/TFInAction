#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:13:52 2018

@author: volvetzhang
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#
#logs_path = 'log_mnist_softmax'
batch_size = 100
learning_rate = 0.1
training_epoch = 20
mnist = input_data.read_data_sets('data', one_hot=True)
##print(mnist.train.images.shape)
##print(mnist.train.labels.shape)
##print(mnist.test.images.shape)
##print(mnist.test.labels.shape)
##im = mnist.train.images[0]
##im = np.resize(im, [28, 28])
##plt.imshow(im, cmap='Greys_r')
##plt.show()
#
X = tf.placeholder(tf.float32, [None, 784], name='input')
Y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#
Ylogits = tf.matmul(X, W) + b
Y = tf.nn.softmax(Ylogits, name='output')
#
#
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Ylogits))
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
#
#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())
    for epoch in range(training_epoch):
        batch_count = int(mnist.train.num_examples/batch_size)
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run([train_step], feed_dict={X: batch_x, Y_: batch_y})
        print("Epoch: ", epoch, ", Accuracy: ", accuracy.eval(feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))
    saver = tf.train.Saver()
    save_path = saver.save(sess, "../model/mnist_simple_softmax.ckpt")
    print('done')