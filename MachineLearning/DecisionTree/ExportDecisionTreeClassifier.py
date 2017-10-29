from sklearn.tree import DecisionTreeClassifier, export_graphviz
import BaseClass
import sys
import os


class ExportDecisionTreeClassifier(BaseClass.Base):
    def __init__(self):
        super(ExportDecisionTreeClassifier, self).__init__()
        self.path = sys.path[0]

    def run(self):
        X_train, X_test, Y_train, Y_test = self.load_datasets_from_sklearn(test_size=0.25,
                                                                           data_set_name="iris",
                                                                           stratify_on_target=True)
        t = DecisionTreeClassifier()
        t.fit(X_train, Y_train)
        file_path = self.path + "/DecisionTreeOnIris"
        export_graphviz(t, file_path)
        os.system("dot -Tpdf {0} -o {1}".format(file_path, file_path + ".pdf"))


if __name__ == "__main__":
    t = ExportDecisionTreeClassifier()
    t.run()
