from sklearn.naive_bayes import GaussianNB
import BaseClass


class GaussianNBOnDigits(BaseClass.Base):
    def __init__(self):
        super(GaussianNBOnDigits, self).__init__()

    def run(self):
        X_train, X_test, Y_train, Y_test = self.load_datasets_from_sklearn(test_size=0.25, data_set_name="digits")
        gaussian = GaussianNB()
        gaussian.fit(X_train, Y_train)
        print "Training score: %.2f" % gaussian.score(X_train, Y_train)
        print "Testing score: %.2f" % gaussian.score(X_test, Y_test)


if __name__ == "__main__":
    t = GaussianNBOnDigits()
    t.run()
