
import tensorflow as tf
import tfutility

class LinearRegression:
    def __init__(self, shape, lamda=0.0000001):
        self.w = tf.Variable(tf.zeros(shape), name="weights")
        self.b = tf.Variable(0., name="bias")
        self.lamda = lamda
        self.sess = tf.Session()
        return

    def __del__(self):
        self.sess.close()

    def inference(self, x):
        return  tf.matmul(x, self.w) + self.b

    def loss(self, x, y):
        y_predicted = self.inference(x)
        return tf.reduce_sum(tf.squared_difference(y, y_predicted))

    def train(self, total_loss):
        return tf.train.GradientDescentOptimizer(self.lamda).minimize(total_loss)

    def run(self, X, Y):
        #self.writer = tf.summary.FileWriter('./lr_graph', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        total_loss = self.loss(X, Y)
        train_op = self.train(total_loss)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        training_step = 1000
        for step in range(training_step):
            self.sess.run(train_op)
            if step % 100 == 0:
                print "Loss: ", self.sess.run(total_loss)
        coord.request_stop()
        coord.join(threads)
        #self.writer.close()
        return

    def evaluate(self, X):
        return self.sess.run(self.inference(X))