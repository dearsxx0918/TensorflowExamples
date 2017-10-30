from sklearn.naive_bayes import BernoulliNB
import BaseClass
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class BernoulliNBOnDigitsWithDifferentAlphasAndDifferentBinarize(BaseClass.Base):
    def __init__(self):
        super(BernoulliNBOnDigitsWithDifferentAlphasAndDifferentBinarize, self).__init__()

    def run(self):
        X_train, X_test, Y_train, Y_test = self.load_datasets_from_sklearn(test_size=0.25, data_set_name="digits")
        alphas = np.logspace(-2, 2, 50)
        min_x = min(np.min(X_train.ravel()), np.min(X_test.ravel())) - 0.1
        max_x = max(np.max(X_train.ravel()), np.max(X_test.ravel())) + 0.1
        binarizes = np.linspace(min_x, max_x, endpoint=True, num=50)
        test_scores = []
        for alpha in alphas:
            for binarize in binarizes:
                bernoulli = BernoulliNB(alpha=alpha, binarize=binarize)
                bernoulli.fit(X_train, Y_train)
                test_scores.append(bernoulli.score(X_test, Y_test))

        alphas, binarizes = np.meshgrid(alphas, binarizes)
        test_scores = np.array(test_scores).reshape(alphas.shape)
        ax = Axes3D(self.fig)
        surf = ax.plot_surface(alphas,
                               binarizes,
                               test_scores,
                               rstride=1,
                               cstride=1,
                               cmap=cm.jet,
                               linewidth=0,
                               antialiased=False)

        self.fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_title("Bernoulli")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("binarize")
        ax.set_zlabel(r"score")


if __name__ == "__main__":
    bernoulli = BernoulliNBOnDigitsWithDifferentAlphasAndDifferentBinarize()
    bernoulli.run()
    bernoulli.show()
