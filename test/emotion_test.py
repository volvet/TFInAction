#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:13:06 2018

@author: volvetzhang

Data Set: https://www.kaggle.com/c/facial-keypoints-detector

"""

import pandas as pd
import numpy as np
import tensorflow as tf
import os
#import matplotlib.pyplot as plt


IMG_SIZE = 48
NUM_LABELS = 7
VALIDATION_PERCENT = 0.1
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_ITERATIONS = 1001
REGULARIZATION = 1e-2

emotions = { 0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 
            5: 'surprise', 6: 'neutral'}

#FLAGS = tf.flags.FLAGS
#tf.flags.DEFINE_string('data_dir', './Emotion/', '')
#tf.flags.DEFINE_string('mode_dir', '../model/', '')


#class TestResult:
#    def __init__(self):
#        self.anger = 0
#        self.disgust = 0
#        self.fear = 0
#        self.happy = 0
#        self.sad = 0
#        self.surprise = 0
#        self.neutral = 0
#    
#    def evaluate(self, label):
#        if 0 == label:
#            self.anger = self.anger + 1
#        if 1 == label:
#            self.disgust = self.disgust + 1
#        if 2 == label:
#            self.fear = self.fear + 1
#        if 3 == label:
#            self.happy = self.happy + 1
#        if 4 == label:
#            self.sad = self.sad + 1
#        if 5 == label:
#            self.surprise = self.surprise + 1
#        if 6 == label:
#            self.neutral = self.neutral + 1
#    
#    def display_result(self,evaluations):
#        print("anger = "    + str((self.anger/float(evaluations))*100)    + "%")
#        print("disgust = "  + str((self.disgust/float(evaluations))*100)  + "%")
#        print("fear = "     + str((self.fear/float(evaluations))*100)     + "%")
#        print("happy = "    + str((self.happy/float(evaluations))*100)    + "%")
#        print("sad = "      + str((self.sad/float(evaluations))*100)      + "%")
#        print("surprise = " + str((self.surprise/float(evaluations))*100) + "%")
#        print("neutral = "  + str((self.neutral/float(evaluations))*100)  + "%")
        
        

def create_emotion_label(x):
    label = np.zeros((1, NUM_LABELS), dtype=np.float32)
    label[:, int(x)] = 1
    return label

def read_data(data_dir):
    train_filename = os.path.join(data_dir, 'train.csv')
    data_frame = pd.read_csv(train_filename)
    #print(data_frame)
    data_frame['Pixels'] = data_frame['Pixels'].apply(lambda x: np.fromstring(x, dtype=float, sep=" ")/255.0)
    #print(data_frame)
    data_frame = data_frame.dropna()
    
    train_images = np.vstack(data_frame['Pixels']).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    #print(train_images.shape)
    train_labels = np.array(list(map(create_emotion_label, data_frame['Emotion'].values))).reshape(-1, NUM_LABELS)
    #print(train_labels.shape)
        
    permutations = np.random.permutation(train_images.shape[0])
    train_images = train_images[permutations]
    train_labels = train_labels[permutations]
        
    validations = int(train_images.shape[0]*VALIDATION_PERCENT)
    validation_images = train_images[:validations]
    validation_labels = train_labels[:validations]
    train_images = train_images[validations:]
    train_labels = train_labels[validations:]
    
    test_filename = os.path.join(data_dir, 'test.csv')
    data_frame = pd.read_csv(test_filename)
    #print(data_frame)
    data_frame['Pixels'] = data_frame['Pixels'].apply(lambda x: np.fromstring(x, dtype=float, sep=" ")/255.0)
    data_frame = data_frame.dropna();
    test_images = np.vstack(data_frame['Pixels']).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return train_images, train_labels, validation_images, validation_labels, test_images


def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    return tf.nn.bias_add(conv, bias)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def add_to_regularization_loss(W, b):
    tf.add_to_collection('losses', tf.nn.l2_loss(W))
    tf.add_to_collection('losses', tf.nn.l2_loss(b))

weights = {
    'wc1': weight_variable([5,5,1,32], name='W_conv1'),
    'wc2': weight_variable([3,3,32,64], name='W_conv2'),
    'wf1': weight_variable([IMG_SIZE//4*IMG_SIZE//4*64, 256], name='W_fc1'),
    'wf2': weight_variable([256, NUM_LABELS], name='W_fc2')
}

biases = {
    'bc1': bias_variable([32], name='b_conv1'),
    'bc2': bias_variable([64], name='b_conv2'),
    'bf1': bias_variable([256], name='b_fc1'),
    'bf2': bias_variable([NUM_LABELS], name='b_fc2')
}


def build_emotion_cnn(dataset):
    with tf.name_scope('conv1') as scope:
        conv1 = conv2d_basic(dataset, weights['wc1'], biases['bc1'])
        conv1 = tf.nn.relu(conv1)
        conv1 = max_pool_2x2(conv1)
        add_to_regularization_loss(weights['wc1'], biases['bc1'])
    
    with tf.name_scope('conv2') as scope:
        conv2 = conv2d_basic(conv1, weights['wc2'], biases['bc2'])
        conv2 = tf.nn.relu(conv2)
        conv2 = max_pool_2x2(conv2)
        add_to_regularization_loss(weights['wc2'], biases['bc2'])
        
    with tf.name_scope('fc_1') as scope:
        prob = 0.5
        h_flat = tf.reshape(conv2, [-1, IMG_SIZE*IMG_SIZE//16*64])
        fc1 = tf.nn.relu(tf.matmul(h_flat, weights['wf1']) + biases['bf1'])
        fc1 = tf.nn.dropout(fc1, prob)
        
    with tf.name_scope('fc_2') as scope:
        pred = tf.matmul(fc1, weights['wf2']) + biases['bf2']
        
    return pred


def loss(pred, label):
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label))
    reg_losses = tf.add_n(tf.get_collection('losses'))
    return cross_entropy_loss + REGULARIZATION * reg_losses

def train(loss, step):
    return tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=step)

def get_next_batch(dataset, labels, step):
    offset = (step * BATCH_SIZE) % (dataset.shape[0] - BATCH_SIZE)
    batch_data = dataset[offset: offset+BATCH_SIZE]
    batch_labels = labels[offset: offset+BATCH_SIZE]
    return batch_data, batch_labels


def main():
    train_images, train_labels, validation_images, validation_labels, _ = read_data('./Emotion/')
    print("Train size: ", train_images.shape[0])
    print("Validation size: ", validation_images.shape[0])
    
    #img = test_images[0]
    #img = np.resize(img, (IMG_SIZE, IMG_SIZE))
    #plt.imshow(img, cmap='Greys_r')
    
    global_step = tf.Variable(0, trainable=False)
    #dropout_prob = tf.placeholder(tf.float32)
    input_dataset = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 1], name='input')
    input_labels = tf.placeholder(tf.float32, [None, NUM_LABELS])
    
    pred = build_emotion_cnn(input_dataset)
    #output_pred = tf.nn.softmax(pred, name='output')
    loss_val = loss(pred, input_labels)
    train_op = train(loss_val, global_step)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for step in range(MAX_ITERATIONS):
            batch_imgs, batch_labels = get_next_batch(train_images, train_labels, step)
            feed_dict = { input_dataset: batch_imgs, input_labels: batch_labels}
            sess.run(train_op, feed_dict=feed_dict)
            
            if step % 10 == 0:
                train_loss = sess.run([loss_val], feed_dict=feed_dict)
                print("Training Loss: ", train_loss)
    
    return


if __name__ == '__main__':
    main()