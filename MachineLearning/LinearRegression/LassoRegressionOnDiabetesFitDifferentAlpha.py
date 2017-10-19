import BaseClass
from sklearn import linear_model


class LassoRegressionOnDiabetesFitDifferentAlpha(BaseClass.Base):
    def __init__(self):
        super(LassoRegressionOnDiabetesFitDifferentAlpha, self).__init__()

    def run(self):
        X_train, X_test, Y_train, Y_test = self.load_datasets_from_sklearn(test_size=0.25)
        alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        scores = []
        for alpha in alphas:
            lasso = linear_model.Lasso(alpha=alpha)
            lasso.fit(X_train, Y_train)
            scores.append(lasso.score(X_test, Y_test))

        ax = self.fig.add_subplot(1, 1, 1)
        ax.set_title("Lasso")
        ax.plot(alphas, scores, marker="o", color="b")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"score")
        ax.set_xscale('log')


if __name__ == "__main__":
    lasso = LassoRegressionOnDiabetesFitDifferentAlpha()
    lasso.run()
    lasso.show()
