import BaseClass
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class SimpleCNN(BaseClass.Base):
    def __init__(self, data_path="Mnist/"):
        super(SimpleCNN, self).__init__()
        self.path = self.base_path + data_path

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def run(self):
        sess = tf.Session()
        mnist = input_data.read_data_sets(self.path, one_hot=True)
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        w_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, w_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        w_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        w_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        w_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        init = tf.global_variables_initializer()

        sess.run(init)

        for i in range(20001):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

            if i % 2000 == 0:
                correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                print "Step : ", i, "  Accuracy :", sess.run(accuracy,
                                                             feed_dict={
                                                                 x: mnist.test.images,
                                                                 y_: mnist.test.labels,
                                                                 keep_prob: 1.0})


if __name__ == "__main__":
    c = SimpleCNN()
    c.run()
