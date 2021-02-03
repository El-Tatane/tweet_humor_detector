from .ModelBase import ModelBase
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score


class XGBoost(ModelBase):

    def __init__(self, num_class, params=None, num_round=10):
        super().__init__()
        self.param = {'max_depth': 2, 'eta': 1, 'nthread': 4} if params is None else params
        self.param["num_class"] = num_class
        self.num_round = num_round

    def fit(self, X, y):
        data = self.convert_data(X, y)
        self.bst = xgb.train(self.param, data, self.num_round)

    def predict(self, X):
        data = self.convert_data(X)
        return self.bst.predict(data)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def convert_data(self, X, y=None):
        if y is not None:
            label = np.array(y)
            data = xgb.DMatrix(np.array(X), label=label)
        else:
            data = xgb.DMatrix(np.array(X))
        return data