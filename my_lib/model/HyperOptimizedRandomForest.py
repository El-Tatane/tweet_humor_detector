from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import pandas as pd
from .ModelBase import ModelBase


class HyperOptimizedRandomForest(ModelBase):

    def __init__(self, param_grid=None):
        super().__init__()
        self.param_grid = {
            'n_estimators': [10, 50, 100, 120, 150, 200],
            'max_depth': [2, 5, 7, 10, 15, 20]
        } if param_grid is None else param_grid


    def fit(self, X, y):
        self.clf = RandomForestClassifier()
        self.grid_clf = GridSearchCV(self.clf, self.param_grid)
        self.grid_clf.fit(X, y)

    def predict(self, X):
        return self.grid_clf.predict(X)

    def score(self, X, y):
        return self.grid_clf.score(X, y)