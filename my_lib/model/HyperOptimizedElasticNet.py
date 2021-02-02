from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from .ModelBase import ModelBase


class HyperOptimizedElasticNet(ModelBase):

    def __init__(self, param_grid=None):
        super().__init__()
        self.param_grid = {
            'l1_ratio': [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]
        } if param_grid is None else param_grid

    def fit(self, X, y):
        self.lr = LogisticRegression(penalty='elasticnet', solver='saga')
        self.grid_clf = GridSearchCV(self.lr, self.param_grid)
        self.grid_clf.fit(X, y)

    def predict(self, X):
        return self.grid_clf.predict(X)

    def score(self, X, y):
        return self.grid_clf.score(X, y)

