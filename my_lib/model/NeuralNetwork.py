from .ModelBase import ModelBase
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np
import os


class NeuralNetwork(ModelBase):

    def __init__(self, architecture: list, num_class, optimizer="adam", loss="categorical_crossentropy"):
        """

        :param architecture: (unit, activation), soft_max activation for last dense

        """

        assert num_class == architecture[-1]["units"]

        super().__init__()
        self.nn = []
        for dense_dict in architecture:
            self.nn.append(tf.keras.layers.Dense(**dense_dict))
        self.num_classes = num_class
        self.model = tf.keras.Sequential(self.nn)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


    def fit(self, X, y, batch_size=120, epoch=90):
        X, y = np.array(X), np.array(y)
        y_vector = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
        self.model.fit(X, y_vector)

    def predict(self, X):
        X = np.array(X)
        return self.model.predict(X)

    def score(self, X, y):
        X, y = np.array(X), np.array(y)
        y_vector_pred = self.predict(X)
        y_pred = np.argmax(y_vector_pred, axis=1)
        return accuracy_score(y, y_pred)

    def save(self, filename):
        path = os.path.join("/app", "models", filename)
        if os.path.isfile(path):
            print("del")
            os.remove(path)

        return self.model.save(path)

    def load(self, filename):
        path = os.path.join("/app", "models", filename)
        self.model = tf.keras.models.load_model(path)