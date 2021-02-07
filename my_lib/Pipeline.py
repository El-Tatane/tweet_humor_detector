from .LabelEncoder import LabelEncoder
from .processing import *
from .model import *
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec


class Pipeline:

    def __init__(self):
        pass

    def load_data(self, is_test=False):
        # lecture fichier
        df_data = pd.read_fwf("/data/train.txt", index_col=False, names=['annot', 'tweet'])

        df_data["label"] = df_data.apply(lambda row: row["annot"].split(",")[1], axis="columns")
        df_data["enterprise"] = df_data.apply(lambda row: row["annot"].split(",")[2][:-1], axis="columns")
        df_data.drop("annot", axis="columns", inplace=True)

        if is_test is True:
            df_data = df_data[0:500]
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

    def get_train_test(self, df, input_col="input_vector", filter_label_list=None, true_output_list=None):

        if filter_label_list is not None:
            df = df.loc[df["label"].isin(filter_label_list), ["label", input_col]]

        train_data, test_data = split_data(df)

        df_X_train = train_data.loc[:, [input_col]]
        if true_output_list is None:
            y_train = train_data["label"].values
        else:
            y_train = np.array([1 if el in true_output_list else 0 for el in train_data["label"].values])

        df_X_test = test_data.loc[:, [input_col]]
        if true_output_list is None:
            y_test = test_data["label"].values
        else:
            y_test = np.array([1 if el in true_output_list else 0 for el in test_data["label"].values])

        X_train = []
        X_test = []
        for idx, row in df_X_train.iterrows():
            X_train.append(row[input_col])

        for idx, row in df_X_test.iterrows():
            X_test.append(row[input_col])

        return np.array(X_train).astype("int"), np.array(X_test).astype("int"), y_train.astype("int"), y_test.astype("int")

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

    def train_embedding_words(self, df, vector_size=100, embedding_type="CBOW", word_max_number=160):
        ref_embedding = {"skip-gram": 1,  "CBOW": 0}
        embedding_type = ref_embedding[embedding_type]

        sentence_list = [sentence.split(" ") for sentence in df["token_list"].values]

        model_skip_gram = Word2Vec(sentences=sentence_list, window=5, size=vector_size, min_count=1, sg=embedding_type)
        df["embedding_vector"] = df.apply(lambda row: [model_skip_gram[word] for word in row["token_list"].split(" ")] + [], axis=1)
        df["embedding_vector"] = df.apply(lambda row: row["embedding_vector"] + [[0] * vector_size] * (word_max_number - len(row["embedding_vector"])), axis=1)

        return df
