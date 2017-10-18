from sklearn import linear_model
import BaseClass


class RedgeLassoComparisonWithDifferentAlpha(BaseClass.Base):
    def __init__(self):
        super(RedgeLassoComparisonWithDifferentAlpha, self).__init__()

    def run(self):
        X_train, X_test, Y_train, Y_test = self.load_datasets_from_sklearn(test_size=0.25)
        alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        scores = {"ridge":[],"lasso":[]}
        for i, alpha in enumerate(alphas):
            ridge = linear_model.Ridge(alpha=alpha)
            ridge.fit(X_train, Y_train)
            scores["ridge"].append(ridge.score(X_test, Y_test))

            lasso = linear_model.Lasso(alpha=alpha)
            lasso.fit(X_train, Y_train)
            scores["lasso"].append(lasso.score(X_test, Y_test))

        ax = self.fig.add_subplot(1, 1, 1)
        ax.set_title("comparison")
        ax.plot(alphas, scores["ridge"], marker="*", color="r")
        ax.plot(alphas, scores["lasso"], marker="o", color="b")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"score")
        ax.set_xscale('log')


if __name__ == "__main__":
    com = RedgeLassoComparisonWithDifferentAlpha()
    com.run()
    com.show()