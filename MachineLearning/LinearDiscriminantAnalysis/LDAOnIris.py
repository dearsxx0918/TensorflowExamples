import numpy as np
import BaseClass
from sklearn import discriminant_analysis
from mpl_toolkits.mplot3d import Axes3D


class LDAOnIris(BaseClass.Base):
    def __init__(self):
        super(LDAOnIris, self).__init__()

    def run(self):
        X_train, X_test, Y_train, Y_test = self.load_datasets_from_sklearn(data_set_name="iris", test_size=0.25)
        shrinkages = np.linspace(0.0, 1.0, num=20)
        scores = []
        for shrinkage in shrinkages:
            lda = discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage)
            lda.fit(X_train, Y_train)
            scores.append(lda.score(X_test, Y_test))
        ax = self.fig.add_subplot(1, 1, 1)
        ax.plot(shrinkages, scores)
        ax.set_xlabel(r"shrinkage")
        ax.set_ylabel(r"score")
        ax.set_ylim(0, 1.05)
        ax.set_title("LinearDiscriminantAnalysisOnDifferentShrinkages")


if __name__ == "__main__":
    lda = LDAOnIris()
    lda.run()
    lda.show()
