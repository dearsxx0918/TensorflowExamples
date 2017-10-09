import tensorflow as tf
import BaseClass
from tensorflow.examples.tutorials.mnist import input_data


class MLPOnMnist(BaseClass.Base):
    def __init__(self, data_path="Mnist/"):
        super(MLPOnMnist, self).__init__()
        self.path = self.base_path + data_path
        print self.base_path
        print self.path

    def run(self):
        sess = tf.Session()
        mnist = input_data.read_data_sets(self.path, one_hot=True)
        n_inputs = 784
        n_hidden1 = 300
        w1 = tf.Variable(tf.truncated_normal([n_inputs, n_hidden1], stddev=0.1))
        b1 = tf.Variable(tf.zeros([n_hidden1]))
        w2 = tf.Variable(tf.zeros([n_hidden1, 10]))
        b2 = tf.Variable(tf.zeros([10]))

        x = tf.placeholder(tf.float32, [None, n_inputs])
        keep_prob = tf.placeholder(tf.float32)
        n_hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)
        hidden_drop = tf.nn.dropout(n_hidden1, keep_prob)
        y = tf.nn.softmax(tf.matmul(hidden_drop, w2) + b2)
        y_ = tf.placeholder(tf.float32, [None, 10])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(3001):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.75})

            if i % 300 == 0:
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                print "Step : ", i, "  Accuracy :", sess.run(accuracy,
                                                             feed_dict={
                                                                 x: mnist.test.images,
                                                                 y_: mnist.test.labels,
                                                                 keep_prob: 1.0})


if __name__ == "__main__":
    a = MLPOnMnist()
    a.run()
