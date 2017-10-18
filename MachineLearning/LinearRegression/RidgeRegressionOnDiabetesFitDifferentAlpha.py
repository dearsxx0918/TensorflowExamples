from sklearn import linear_model
import BaseClass


class RidgeRegressionOnDiabetesFitDifferentAlpha(BaseClass.Base):
    def __init__(self):
        super(RidgeRegressionOnDiabetesFitDifferentAlpha, self).__init__()

    def run(self):
        X_train, X_test, Y_train, Y_test = self.load_datasets_from_sklearn(test_size=0.25)
        alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        scores = []
        for alpha in alphas:
            ridge = linear_model.Ridge(alpha=alpha)
            ridge.fit(X_train, Y_train)
            scores.append(ridge.score(X_test, Y_test))

        ax = self.fig.add_subplot(1, 1, 1)
        ax.set_title("Ridge")
        ax.plot(alphas, scores, marker="*", color="r")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"score")
        ax.set_xscale('log')

if __name__ == "__main__":
    redge = RidgeRegressionOnDiabetesFitDifferentAlpha()
    redge.run()
    redge.show()