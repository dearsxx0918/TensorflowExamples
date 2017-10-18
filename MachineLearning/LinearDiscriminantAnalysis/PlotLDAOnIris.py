import numpy as np
import BaseClass
from sklearn import discriminant_analysis
from mpl_toolkits.mplot3d import Axes3D
import warnings


class PlotLDAOnIris(BaseClass.Base):
    def __init__(self):
        super(PlotLDAOnIris, self).__init__()

    def run(self):
        warnings.filterwarnings("ignore")
        X_train, X_test, Y_train, Y_test = self.load_datasets_from_sklearn(data_set_name="iris", test_size=0.25)
        X = np.vstack((X_train, X_test))
        Y = np.vstack((Y_train.reshape(Y_train.size, 1), Y_test.reshape(Y_test.size, 1)))
        lda = discriminant_analysis.LinearDiscriminantAnalysis()
        Y = Y.ravel()
        lda.fit(X, Y)
        converted_X = np.dot(X, np.transpose(lda.coef_)) + lda.intercept_
        ax = Axes3D(self.fig)
        colors = 'rgb'
        markers = 'o*s'
        for target, color, marker in zip([0, 1, 2], colors, markers):
            pos = (Y == target).ravel()
            x = converted_X[pos, :]
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], color=color, marker=marker, label=('Label %d' % target))
            ax.legend(loc="best")
            self.fig.suptitle("Iris after LDA")


if __name__ == "__main__":
    lda = PlotLDAOnIris()
    lda.run()
    lda.show()
