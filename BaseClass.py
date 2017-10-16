import sys
from sklearn import datasets, model_selection

class Base(object):
    def __init__(self):
        self._path = None
        base_path_str = sys.path[0]
        index = base_path_str.find("TensorflowExamples") + len("TensorflowExamples")
        self.base_path = base_path_str[: index] + "/DataSets/"

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if isinstance(value, str):
            self._path = value
        else:
            raise Exception("Path should be a string.")

    @staticmethod
    def load_datasets_from_sklearn(test_size=0.3, random_state=0, datasets_name="diabetes"):
        data = getattr(datasets, "load_" + datasets_name)()
        return model_selection.train_test_split(data.data, data.target, test_size=test_size, random_state=random_state)

    def run(self):
        pass


if __name__ == "__main_":
    b = Base()
    b.run()
