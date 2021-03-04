from . import ModelBase
from transformers import pipeline
from sklearn.metrics import accuracy_score
import numpy as np


class BERT(ModelBase):

    def __init__(self):
        super().__init__()
        self.classifier = pipeline("sentiment-analysis")

    def fit(self, X, y):
        print("already fit")
        return self

    def predict(self, X):
        """

        Args:
            X (str or string list): string sentence

        Returns:

        """
        res: dict = self.classifier(X)
        ref = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
        return [ref.index(el_dict["label"]) for el_dict in res]

    def score(self, X, y):

        y_pred = []
        for i, X_el in enumerate(X):
            if i % 100 == 0:
                print(i, end="")
            res: int = self.predict(X_el)[0]

            y_pred.append(res)
        y_pred = np.array(y_pred)

        return accuracy_score(list(y), list(y_pred))
