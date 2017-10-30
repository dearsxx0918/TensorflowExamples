import numpy as np
from sklearn.naive_bayes import MultinomialNB
import BaseClass


class MultinomialNBOnDigitsWithDifferentAlphas(BaseClass.Base):
    def __init__(self):
        super(MultinomialNBOnDigitsWithDifferentAlphas, self).__init__()

    def run(self):
        alphas = np.logspace(-2, 5, num=200)
        X_train, X_test, Y_train, Y_test = self.load_datasets_from_sklearn(test_size=0.25,
                                                                           data_set_name="digits")
        train_scores = []
        test_scores = []
        for alpha in alphas:
            multi = MultinomialNB(alpha=alpha)
            multi.fit(X_train, Y_train)
            train_scores.append(multi.score(X_train, Y_train))
            test_scores.append(multi.score(X_test, Y_test))
        ax = self.fig.add_subplot(1, 1, 1)
        ax.plot(alphas, train_scores, label="Train scores", color="b")
        ax.plot(alphas, test_scores, label="Test scores", color="r")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("score")
        ax.set_ylim(0.0, 1.0)
        ax.set_xscale("log")
        ax.set_title("Multinomial naive beyes classifier with different alpha")


if __name__ == "__main__":
    multi = MultinomialNBOnDigitsWithDifferentAlphas()
    multi.run()
    multi.show()
