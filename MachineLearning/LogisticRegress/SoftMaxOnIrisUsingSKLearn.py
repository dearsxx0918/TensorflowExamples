import BaseClass
from sklearn import linear_model
import numpy as np


class SoftMaxOnIris(BaseClass.Base):
    def __init__(self):
        super(SoftMaxOnIris, self).__init__()

    def run(self):
        X_train, X_test, Y_train, Y_test = self.load_datasets_from_sklearn(data_set_name="iris",
                                                                           test_size=0.25,
                                                                           stratify_on_target=True)

        Cs = np.logspace(-2, 4, num=100)
        scores = []
        for C in Cs:
            logistic = linear_model.LogisticRegression(C=C, solver="lbfgs", multi_class='multinomial')
            logistic.fit(X_train, Y_train)
            scores.append(logistic.score(X_test, Y_test))

        ax = self.fig.add_subplot(1, 1, 1)
        ax.set_title("Logistic Regression")
        ax.plot(Cs, scores, color="b", marker="*")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"score")
        ax.set_xscale('log')


if __name__ == "__main__":
    softmax = SoftMaxOnIris()
    softmax.run()
    softmax.show()
