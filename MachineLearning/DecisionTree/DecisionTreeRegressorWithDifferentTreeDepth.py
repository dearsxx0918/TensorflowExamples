import numpy as np
from sklearn.tree import DecisionTreeRegressor
import  BaseClass

class DecisionTreeRegressorWithDifferentTreeDepth(BaseClass.Base):
    def __init__(self):
        super(DecisionTreeRegressorWithDifferentTreeDepth, self).__init__()

    def run(self):
        data_size = 10000
        maxdepth = 50
        depths = np.arange(1, maxdepth)
        X_train, X_test, Y_train, Y_test = self.create_random_data(size=data_size, dimension=1, curve="sin")
        train_scores = []
        test_scores = []
        for depth in depths:
            t = DecisionTreeRegressor(max_depth=depth)
            t.fit(X_train, Y_train)
            train_scores.append(t.score(X_train, Y_train))
            test_scores.append(t.score(X_test, Y_test))
        ax = self.fig.add_subplot(1, 1, 1)
        ax.plot(depths, train_scores, label="Train scores", color="g")
        ax.plot(depths, test_scores, label="Test scores", color="r")
        ax.set_xlabel("max depth")
        ax.set_ylabel("score")
        ax.set_title("Decision tree regressor with different depth")


if __name__ == "__main__":
    t = DecisionTreeRegressorWithDifferentTreeDepth()
    t.run()
    t.show()
