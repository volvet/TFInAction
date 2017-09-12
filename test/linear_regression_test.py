import sys
import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src import linear_regression

def linear_regression_test():
    lr = linear_regression.LinearRegression([2, 1])
    X = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    Y = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181, 274, 303, 244]
    lr.run(tf.to_float(X), tf.to_float(Y))

    #print lr.evaluate([[80., 25.]])
    #print lr.evaluate([[65., 25.]])
    #print lr.evaluate(tf.to_float(X)).transpose()
    #print Y
    print "linear_regression_test:      [OK]"
    return