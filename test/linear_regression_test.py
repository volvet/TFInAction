import sys
import os
import tensorflow as tf
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src import linear_regression


def linear_regression_test():
    print "[Run] linear_regression_test"
    lr = linear_regression.LinearRegression([2, 1])
    X = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44], [27, 24],
         [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], [59, 46], [67, 23],
         [85, 37], [55, 40], [63, 30]]
    Y = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181,
         274, 303, 244]
    lr.run(tf.to_float(X), tf.to_float(Y), 1000)

    # print lr.evaluate([[80., 25.]])
    # print lr.evaluate([[65., 25.]])
    # print lr.evaluate(tf.to_float(X)).transpose()
    # print Y
    print " [OK] linear_regression_test"
    return


def linear_regression2_test():
    print "[Run] linear_regression2_test"
    lr = linear_regression.LinearRegression2([2])
    X = [[84., 46.], [73., 20.], [65., 52.], [70., 30.], [76., 57.], [69., 25.], [63., 28.], [72., 36.], [79., 57.], [75., 44.], [27., 24.],
         [89., 31.], [65., 52.], [57., 23.], [59., 60.], [69., 48.], [60., 34.], [79., 51.], [75., 50.], [82., 34.], [59., 46.], [67., 23.],
         [85., 37.], [55., 40.], [63., 30.]]
    Y = [354., 190., 405., 263., 451., 302., 288., 385., 402., 365., 209., 290., 346., 254., 395., 434., 220., 374., 308., 220., 311., 181.,
         274., 303., 244.]

    #X = np.array([1., 2., 3., 4.])
    #Y = np.array([0., -1., -2., -3.])
    lr.run(np.array(X), np.array(Y), 25, 1000)

    e_X = [[50., 20.], [50., 70.], [90., 20.], [90., 70.]]
    e_Y = [ 303., 256., 303., 256. ]
    print lr.evaluate(np.array(e_X), np.array(e_Y), 4)

    print " [OK] linear_regression_test2"
    return
