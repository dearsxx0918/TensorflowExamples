import BaseClass
import sklearn.preprocessing as prep
import tensorflow.examples.tutorials.mnist.input_data as input_data
from AutoEncoder.AutoEncoder import *
import numpy as np


class AutoEncoderOnMnist(BaseClass.Base):
    def __init__(self, data_path="Mnist/"):
        super(AutoEncoderOnMnist, self).__init__()
        self.path = self.base_path + data_path

    @staticmethod
    def standard_scale(x_train, x_test):
        preprocessor = prep.StandardScaler().fit(x_train)
        x_train = preprocessor.transform(x_train)
        x_test = preprocessor.transform(x_test)
        return x_train, x_test

    @staticmethod
    def get_random_block_from_data(data, batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]

    def run(self):

        auto_encoder = AdditiveGaussionNoiseAutoEncoder(n_input=784,
                                                        n_hidden=200,
                                                        transfer_function=tf.nn.softplus,
                                                        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                                        scale=0.01)
        mnist = input_data.read_data_sets(self.path, one_hot=True)
        n_samples = int(mnist.train.num_examples)
        X_train, X_test = self.standard_scale(mnist.train.images, mnist.test.images)
        training_epochs = 20
        batch_size = 128
        display_step = 1
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(n_samples / batch_size)
            for i in range(total_batch):
                batch_xs = self.get_random_block_from_data(X_train, batch_size)

                cost = auto_encoder.partial_fit(batch_xs)
                avg_cost += cost / n_samples * batch_size

            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Total cost:" + str(auto_encoder.calc_total_cost(X_test)))


if __name__ == "__main__":
    encoder = AutoEncoderOnMnist()
    encoder.run()










