import numpy as np
from sklearn.tree import DecisionTreeRegressor
import BaseClass


class DecisionTreeRegressorWithRandomData(BaseClass.Base):
    def __init__(self):
        super(DecisionTreeRegressorWithRandomData, self).__init__()

    def run(self):
        data_size = 100
        X_train, X_test, Y_train, Y_test = self.create_random_data(size=data_size, dimension=1, curve="sin")
        t = DecisionTreeRegressor()

        t.fit(X_train, Y_train)
        ax = self.fig.add_subplot(1, 1, 1)
        x = np.arange(0, 5.0, 0.01)[:, np.newaxis]
        y = t.predict(x)
        ax.scatter(X_train, Y_train, label="Train samples", color="g")
        ax.scatter(X_test, Y_test, label="Test samples", color="r")
        ax.plot(x, y, label="Predict value", linewidth=2, alpha=0.5)
        ax.set_xlabel("input")
        ax.set_ylabel("output")
        ax.set_title("Decision tree regressor")
        ax.legend(framealpha=0.5)


if __name__ == "__main__":
    t = DecisionTreeRegressorWithRandomData()
    t.run()
    t.show()
