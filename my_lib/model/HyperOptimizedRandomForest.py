from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import pandas as pd
from .ModelBase import ModelBase
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings


class HyperOptimizedRandomForest(ModelBase):

    def __init__(self, param_grid=None):
        super().__init__()
        self.param_grid = {
            'n_estimators': [10, 50, 100, 150, 200],
            'max_depth': [2, 5, 7, 10]
        } if param_grid is None else param_grid

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.clf = RandomForestClassifier()
            self.grid_clf = GridSearchCV(self.clf, self.param_grid)
            self.grid_clf.fit(X, y)

    def predict(self, X):
        return self.grid_clf.predict(X)

    def score(self, X, y):
        return self.grid_clf.score(X, y)