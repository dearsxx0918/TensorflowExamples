from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import BaseClass


class SoftMaxOnMNIST(BaseClass.Base):

    def __init__(self, data_path="Mnist/"):
        super(SoftMaxOnMNIST, self).__init__()
        self._path = self.base_path + data_path


    def run(self):
        mnist = input_data.read_data_sets(self.path, one_hot=True)

        x = tf.placeholder(tf.float32, [None, 784])
        w = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.nn.softmax(tf.matmul(x, w) + b)
        y_ = tf.placeholder(tf.float32, [None, 10])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        for i in range(1001):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            if i % 100 == 0:
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print "Step : ", i, "  Accuracy :", sess.run(accuracy,
                                                             feed_dict={x: mnist.test.images, y_: mnist.test.labels})


if __name__ == "__main__":
    softmax = SoftMaxOnMNIST()
    softmax.run()
