from .ModelBase import ModelBase
from sklearn.svm import SVC


class SVM(ModelBase):

    def __init__(self):
        super().__init__()
        self.model = SVC()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)