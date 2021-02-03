from .LabelEncoder import LabelEncoder
from .processing import *
from .model import *
import pandas as pd
import numpy as np


class Pipeline:

    def __init__(self):
        pass

    def load_data(self):
        # lecture fichier
        df_data = pd.read_fwf("/data/train.txt", index_col=False, names=['annot', 'tweet'])

        df_data["label"] = df_data.apply(lambda row: row["annot"].split(",")[1], axis="columns")
        df_data["enterprise"] = df_data.apply(lambda row: row["annot"].split(",")[2][:-1], axis="columns")
        df_data.drop("annot", axis="columns", inplace=True)
        return df_data

    def encode_to_category(self, df_data):
        self.label_encoder = LabelEncoder()
        df_data["label"] = self.label_encoder.encode(df_data["label"], "label")
        df_data["enterprise"] = self.label_encoder.encode(df_data["enterprise"], "enterprise")
        df_data["is_irrelevant"] = df_data.apply(lambda row: 1 if row["label"] == self.label_encoder.label.index("irr") else 0, axis=1)
        return df_data

    def tokenize(self, df_data, token_list=None):
        self.tokenizer = Tokenizer(df_data)
        df_data = self.tokenizer.fit(token_list)
        return df_data

    def get_train_test(self, df):
        df = remove_useless_col(df, keep_col_list=["label", "input_vector"])
        train_data, test_data = split_data(df)

        df_X_train = train_data.loc[train_data["label"] > 1, ["input_vector"]]
        y_train = np.array([int(el) for el in train_data[train_data["label"] > 1]["label"].values])
        df_X_test = test_data.loc[test_data["label"] > 1, ["input_vector"]]
        y_test = np.array([int(el) for el in test_data[test_data["label"] > 1]["label"].values])

        X_train = []
        X_test = []
        for idx, row in df_X_train.iterrows():
            X_train.append(row["input_vector"])

        for idx, row in df_X_test.iterrows():
            X_test.append(row["input_vector"])

        return np.array(X_train), np.array(X_test), y_train, y_test

    def train_model(self, model_dict, X_train, y_train, X_test=None, y_test=None):
        referential_model_dict = {"adaboost": AdaBoostClassifier, "elastic": HyperOptimizedElasticNet,
                                  "rf": HyperOptimizedRandomForest, "nn": NeuralNetwork, "svm": SVM, "xgboost": XGBoost}
        assert set(model_dict).issubset(set(referential_model_dict)), "Error in model_dict train_model"

        res = {}
        for model_name, model_param_dict in model_dict.items():
            print(f"train model: {model_name}")
            res[model_name] = referential_model_dict[model_name](**model_param_dict)
            res[model_name].fit(X_train, y_train)
            train_score = res[model_name].score(X_train, y_train)
            print(f"train score: {round(train_score, 5)}")
            if X_test is not None:
                test_score = res[model_name].score(X_test, y_test)
                print(f"test score: {round(test_score, 5)}")
            print("")

