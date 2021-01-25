import re
import demoji
import numpy as np
import pandas as pd
demoji.download_codes()

class Tokenizer:

    split_regex = [".", ",", "!", "?", "'", '"', ";", ":"]

    def __init__(self, df, params_dict=None, input_col="tweet", output_col="token_list"):

        assert input_col in df, f"Column {input_col} didn't exist in df"
        assert len(df), "No data in df"

        self.df = df.copy()
        self.params_dict = params_dict if params_dict is not None else {}
        self.input_col = input_col
        self.output_col = output_col



    def fit(self):
        self.prepare()
        self.to_number_vector()
        return self.df


    def prepare(self):

        self.df[self.output_col] = self.df[self.input_col]

        for idx, row in self.df.iterrows():
            row[self.output_col] = self.token_smiley(row[self.output_col])
            row[self.output_col] = self.token_url(row[self.output_col])

            for char in Tokenizer.split_regex:
                row[self.output_col] = row[self.output_col].replace(char, f" {char} ")

            row[self.output_col] = row[self.output_col].split(" ")
            row[self.output_col] = [el for el in row[self.output_col] if el != ""]



    def token_url(self, input_str):
        email_regex = r"(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*))|(bit\.ly\/[^ ]*)"
        return re.sub(email_regex, " <TOKEN_URL> ", input_str)

    # def token_percentage(self, str):

    def token_smiley(self, input_str):
        emoji_dict = demoji.findall(input_str)

        for emoji_img, emoji_txt in emoji_dict.items():
            token_emoji_str = f" <TOKEN_EMOJI_{emoji_txt.replace(' ', '_')}> ".upper()
            input_str = input_str.replace(emoji_img, token_emoji_str)
        return input_str

    def to_number_vector(self, number_vector_col="number_vector"):
        # recuperer la liste du vocabulaire et associé un id
        vocab_set = set()
        for idx, row in self.df.iterrows():
            for word in row[self.output_col]:
                vocab_set.add(word)

        self.df[number_vector_col] = np.nan
        self.df[number_vector_col] = self.df[number_vector_col].astype("object")
        # remplacer chaque vocabulaire par un id
        self.vocab_list = list(vocab_set)
        for idx, row in self.df.iterrows():
            self.df.at[idx, number_vector_col] = []
            for word in row[self.output_col]:
                row[number_vector_col].append(self.vocab_list.index(word))


