#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 13:58:35 2018

@author: volvetzhang
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data', one_hot=True)

learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

tf.reset_default_graph()

X = tf.placeholder('float', [None, n_steps, n_input])
Y = tf.placeholder('float' ,[None, n_classes])

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}

biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


#x = tf.transpose(x, [1, 0, 2])
#x = tf.reshape(x, [-1, n_input])
#x = tf.split(axis=0, num_or_size_splits=n_steps, value=x)
x = tf.unstack(X, n_steps, 1)
lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
output, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
pred = tf.matmul(output[-1], weights['out']) + biases['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
            loss = sess.run(cost, feed_dict={X: batch_x, Y: batch_y})
            print("Step: ", step, ", loss: ", loss, ", acc: ", acc)
        step += 1