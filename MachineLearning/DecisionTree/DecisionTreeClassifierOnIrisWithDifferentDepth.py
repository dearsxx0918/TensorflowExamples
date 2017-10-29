import numpy as np
from sklearn.tree import DecisionTreeClassifier
import  BaseClass

class DecisionTreeClassifierOnIrisWithDifferentTreeDepth(BaseClass.Base):
    def __init__(self):
        super(DecisionTreeClassifierOnIrisWithDifferentTreeDepth, self).__init__()

    def run(self):
        data_size = 10000
        maxdepth = 50
        depths = np.arange(1, maxdepth)
        X_train, X_test, Y_train, Y_test = self.load_datasets_from_sklearn(test_size=0.25,
                                                                           data_set_name="iris",
                                                                           stratify_on_target=True)
        train_scores = []
        test_scores = []
        for depth in depths:
            t = DecisionTreeClassifier(max_depth=depth)
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
    t = DecisionTreeClassifierOnIrisWithDifferentTreeDepth()
    t.run()
    t.show()
