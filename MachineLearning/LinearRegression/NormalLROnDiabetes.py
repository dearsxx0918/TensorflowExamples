import numpy as np
from sklearn import linear_model
import BaseClass


class NormalLROnDiabetes(BaseClass.Base):
    def __index__(self, data_path=None):
        super(NormalLROnDiabetes, self).__init__()
        self.path = self.base_path + data_path

    def run(self):
        X_train, X_test, Y_train, Y_test = self.load_datasets_from_sklearn(test_size=0.25)
        regr = linear_model.LinearRegression()
        regr.fit(X_train, Y_train)
        print('Coefficients:%s, intercept %.2f' % (regr.coef_, regr.intercept_))
        print('Residual sum of squares:%.2f' % np.mean((regr.predict(X_test) - Y_test) ** 2))
        print('Score:%.2f' % regr.score(X_test, Y_test))


if __name__ == "__main__":
    lr = NormalLROnDiabetes()
    lr.run()
