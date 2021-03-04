import dill
import os


class ModelBase:

    def __init__(self):
        self.model = None

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def score(self, X, y):
        raise NotImplementedError()

    def save(self, filename):
        return dill.dump(self.model, open(os.path.join("/app", "models", filename), "wb"))

    def load(self, filename):
        self.model = dill.load(open(os.path.join("/app", "models", filename), "wb"))
