import sys


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

    def run(self):
        pass


if __name__ == "__main_":
    b = Base()
    b.run()
