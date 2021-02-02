from .ModelBase import ModelBase
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score


class Xgboost:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.param = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softmax', "num_class": 4, 'nthread': 4}
        self.y_train = y_train
        self.y_test = y_test
        self.label_train = np.array(y_train)
        self.dtrain = xgb.DMatrix(np.array(X_train), label=self.label_train)

        self.label_test = np.array(y_test)
        self.dtest = xgb.DMatrix(np.array(X_test), label=self.label_test)

        self.evallist = [(self.dtest, 'eval'), (self.dtrain, 'train')]
        self.num_round = 10

    def fit(self):
        self.bst = xgb.train(self.param, self.dtrain, self.num_round, self.evallist)

    def predict(self):
        self.y_train_pred = self.bst.predict(self.dtrain)
        self.y_test_pred  = self.bst.predict(self.dtest)
        return self.y_train_pred, self.y_test_pred

    def score(self):
        return accuracy_score(self.y_train, self.y_train_pred), \
               accuracy_score(self.y_test, self.y_test_pred)

