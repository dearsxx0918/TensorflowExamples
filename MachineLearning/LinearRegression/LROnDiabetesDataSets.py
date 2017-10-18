import BaseClass
import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis
from sklearn.model_selection import train_test_split


class LROnDiabetesDataSets(BaseClass.Base):
    def __init__(self, data_path=""):
        super(LROnDiabetesDataSets, self).__init__()
        self.path = self.base_path + data_path

    @staticmethod
    def load_data():
        diabetes = datasets.load_diabetes()
        return train_test_split(datasets.data, diabetes.target, test_size=0.3, random_state=0)


if __name__ == "__main__":
    lr = LROnDiabetesDataSets()
