from sklearn import linear_model
import BaseClass
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class ElasticNetOnDiabetesFitDifferentAlphaAndRhos(BaseClass.Base):
    def __init__(self):
        super(ElasticNetOnDiabetesFitDifferentAlphaAndRhos, self).__init__()

    def run(self):
        X_train, X_test, Y_train, Y_test = self.load_datasets_from_sklearn(test_size=0.25)
        alphas = np.logspace(-2, 2)
        rhos = np.linspace(0.01, 1)
        scores = []
        for alpha in alphas:
            for rho in rhos:
                elastic = linear_model.ElasticNet(alpha=alpha, l1_ratio=rho)
                elastic.fit(X_train, Y_train)
                scores.append(elastic.score(X_test, Y_test))

        alphas, rhos = np.meshgrid(alphas, rhos)
        scores = np.array(scores).reshape(alphas.shape)
        ax = Axes3D(self.fig)
        surf = ax.plot_surface(alphas, rhos, scores, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
        self.fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_title("ElasticNet")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\rho$")
        ax.set_zlabel(r"score")


if __name__ == "__main__":
    elastic = ElasticNetOnDiabetesFitDifferentAlphaAndRhos()
    elastic.run()
    elastic.show()
