import sys
from sklearn import datasets, model_selection


class Base(object):
    def __init__(self):
        self._path = None
        base_path_str = sys.path[0]
        index = base_path_str.find("TensorflowExamples") + len("TensorflowExamples")
        self.base_path = base_path_str[: index] + "/DataSets/"
        try:
            import matplotlib.pyplot as plt
            self.fig = plt.figure()
        except Exception as e:
            print e.message

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
    def load_datasets_from_sklearn(test_size=0.3,
                                   random_state=0,
                                   data_set_name="diabetes",
                                   stratify_on_target=False,
                                   stratify_on_data=False):
        data = getattr(datasets, "load_" + data_set_name)()
        target = data.target
        data = data.data
        stratify = None
        if stratify_on_data:
            stratify = data
        elif stratify_on_target:
            stratify = target
        return model_selection.train_test_split(data,
                                                target,
                                                test_size=test_size,
                                                random_state=random_state,
                                                stratify=stratify)

    def run(self):
        pass

    def show(self):
        try:
            import matplotlib.pyplot as plt
            plt.show(self.fig)
        except Exception as e:
            print e.message


if __name__ == "__main__":
    b = Base()