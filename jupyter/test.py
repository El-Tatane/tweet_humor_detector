
import pandas as pd
import my_lib
from official.nlp import bert

pipeline = my_lib.Pipeline()
raw_df = pipeline.load_data()
raw_df = pipeline.encode_to_category(raw_df)
raw_df = pipeline.tokenize(raw_df)
raw_df.head()


df = raw_df.loc[raw_df.label.isin([0,1,3,2])]
df = my_lib.processing.remove_useless_col(df, keep_col_list=["label", "clean_tweet"])
df_train_data, df_test_data = my_lib.processing.split_data(df)
print(f"SHAPE -> train : {df_train_data.shape}, test : {df_test_data.shape}")
df.head()

tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
     do_lower_case=True)