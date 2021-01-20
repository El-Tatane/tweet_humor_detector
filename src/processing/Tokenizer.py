import re
import demoji
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
        return self.df


    def prepare(self):

        self.df[self.output_col] = self.df[self.input_col]

        for idx, row in self.df.iterrows():
            row[self.output_col] = self.token_smiley(row[self.output_col])
            # row[self.output_col] = self.token_url(row[self.output_col])

            for char in Tokenizer.split_regex:
                row[self.output_col] = row[self.output_col].replace(char, f" {char} ")

            row[self.output_col] = row[self.output_col].split(" ")
            row[self.output_col] = [el for el in row[self.output_col] if el != ""]



    def token_url(self, input_str):
        email_regex = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
        return re.sub(email_regex, " <TOKEN_URL> ", input_str)

    # def token_percentage(self, str):

    def token_smiley(self, input_str):
        emoji_dict = demoji.findall(input_str)

        for emoji_img, emoji_txt in emoji_dict.items():
            token_emoji_str = f" <TOKEN_EMOJI_{emoji_txt.replace(' ', '_')}> ".upper()
            input_str = input_str.replace(emoji_img, token_emoji_str)
        return input_str

