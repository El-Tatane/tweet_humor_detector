from .LabelEncoder import LabelEncoder
from .processing.Tokenizer import Tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split


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
        # df_data["is_irrelevant"] = df_data.apply(lambda row: 1 if row["label"] == 3 else 0, axis=1)
        return df_data

    def tokenize(self, df_data):
        self.tokenizer = Tokenizer(df_data)
        df_data = self.tokenizer.fit()
        return df_data

    def split_data(self, df_data, test_size=0.2, shuffle=False, random_state=None):
        train_data, test_data = train_test_split(df_data, test_size=test_size, shuffle=shuffle, random_state=random_state)
        return train_data, test_data

    def remove_useless_col(self, df_data, keep_col_list):
        return df_data.copy()[keep_col_list]



