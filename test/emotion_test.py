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

emotions = { 0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 
            5: 'surprise', 6: 'neutral'}


class TestResult:
    def __init__(self):
        self.anger = 0
        self.disgust = 0
        self.fear = 0
        self.happy = 0
        self.sad = 0
        self.surprise = 0
        self.neutral = 0
    
    def evaluate(self, label):
        if 0 == label:
            self.anger = self.anger + 1
        if 1 == label:
            self.disgust = self.disgust + 1
        if 2 == label:
            self.fear = self.fear + 1
        if 3 == label:
            self.happy = self.happy + 1
        if 4 == label:
            self.sad = self.sad + 1
        if 5 == label:
            self.surprise = self.surprise + 1
        if 6 == label:
            self.neutral = self.neutral + 1
    
    def display_result(self,evaluations):
        print("anger = "    + str((self.anger/float(evaluations))*100)    + "%")
        print("disgust = "  + str((self.disgust/float(evaluations))*100)  + "%")
        print("fear = "     + str((self.fear/float(evaluations))*100)     + "%")
        print("happy = "    + str((self.happy/float(evaluations))*100)    + "%")
        print("sad = "      + str((self.sad/float(evaluations))*100)      + "%")
        print("surprise = " + str((self.surprise/float(evaluations))*100) + "%")
        print("neutral = "  + str((self.neutral/float(evaluations))*100)  + "%")
        
        

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


def main():
    test_images, test_labels, validation_images, validation_labels, _ = read_data('./Emotion/')
    print("Train size: ", test_images.shape[0])
    print("Validation size: ", validation_images.shape[0])
    
    #img = test_images[0]
    #img = np.resize(img, (IMG_SIZE, IMG_SIZE))
    #plt.imshow(img, cmap='Greys_r')
    
    return


if __name__ == '__main__':
    main()