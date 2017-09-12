
import tensorflow as tf

class LinearRegression:
    def __init__(self, shape, lamda=0.0000001):
        self.w = tf.variable(tf.zeros(shape), name="weights")
        self.b = tf.variable(0, name="bias")
        self.lamda = lamda
        return

    def inference(self, x):
        return  tf.matmul(x, self.w) + self.b

    def loss(self, x, y):
        y_predicted = self.inference(x)
        return tf.reduce_sum(tf.squread_difference(y, y_predicted))

    def train(self, total_loss):
        return tf.train.GradientDescentOptimizer(self.lamda).minimize(total_loss)

    def run(self, X, Y):
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            total_loss = self.loss(X, Y)
            train_op = self.train(total_loss)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            training_step = 1000
            for step in range(training_step):
                sess.run(train_op)
                if step % 10 == 0:
                    print "Loss: ", sess.run(total_loss)
            coord.request_stop()
            coord.join(threads)
            sess.close()

        return