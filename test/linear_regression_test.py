import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src import linear_regression


def build_test_set():
    w = 2.5
    b = 4.
    x = np.linspace(0, 3, 1000)
    y = x * w + b + np.random.rand(*x.shape)*0.33
    return x, y


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
    print "    [OK]"
    return


def linear_regression2_test():
    print "[Run] linear_regression2_test"

    #lr = linear_regression.LinearRegression2([2])
    #X = [[84., 46.], [73., 20.], [65., 52.], [70., 30.], [76., 57.], [69., 25.], [63., 28.], [72., 36.], [79., 57.], [75., 44.], [27., 24.],
    #     [89., 31.], [65., 52.], [57., 23.], [59., 60.], [69., 48.], [60., 34.], [79., 51.], [75., 50.], [82., 34.], [59., 46.], [67., 23.],
    #     [85., 37.], [55., 40.], [63., 30.]]
    #Y = [354., 190., 405., 263., 451., 302., 288., 385., 402., 365., 209., 290., 346., 254., 395., 434., 220., 374., 308., 220., 311., 181.,
    #     274., 303., 244.]

    #X = np.array([1., 2., 3., 4.])
    #Y = np.array([0., -1., -2., -3.])
    #lr.run(np.array(X), np.array(Y), 25, 1000)

    #e_X = [[50., 20.], [50., 70.], [90., 20.], [90., 70.]]
    #e_Y = [ 303., 256., 303., 256. ]
    #print lr.evaluate(np.array(e_X), np.array(e_Y), 4)

    X, Y = build_test_set()

    lr = linear_regression.LinearRegression2([1])
    lr.run(np.array(X), np.array(Y), 5, 1000)

    print lr.evaluate(X, Y, 1000)

    print "    [OK]"
    return

def linear_regression3_test():
    num_points = 1000
    vectors_set = []
    for i in xrange(num_points):
        x = np.random.normal(0.0, 0.55)
        y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
        vectors_set.append([x, y])
    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]


    w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = w*x_data + b
    loss = tf.reduce_mean(tf.square(y-y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for step in xrange(10000):
        sess.run(train)
        if step % 100 == 0:
            print step, sess.run(w), sess.run(b)

    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(w)*x_data + sess.run(b))
    plt.legend()
    plt.show()
    return