import pandas as pd
import my_lib

pipeline = my_lib.Pipeline()
raw_df = pipeline.load_data()

# DEBUG
# raw_df = raw_df[0:500]

raw_df = pipeline.encode_to_category(raw_df)
raw_df = pipeline.tokenize(raw_df)
raw_df.head()



# df = pipeline.remove_useless_col(raw_df, keep_col_list=["is_irrelevant", "input_vector"])
df = pipeline.remove_useless_col(raw_df, keep_col_list=["label", "input_vector"])
train_data, test_data = pipeline.split_data(df)
print(f"SHAPE -> train : {train_data.shape}, test : {test_data.shape}")

df_X_train = train_data.loc[train_data["label"] > 1, ["input_vector"]]
y_train = [int(el) for el in train_data[train_data["label"] > 1]["label"].values]
df_X_test = test_data.loc[test_data["label"] > 1, ["input_vector"]]
y_test = [int(el) for el in test_data[test_data["label"] > 1]["label"].values]

X_train = []
X_test = []
for idx, row in df_X_train.iterrows():
    X_train.append(row["input_vector"])

for idx, row in df_X_test.iterrows():
    X_test.append(row["input_vector"])

svm_model = my_lib.model.SVM()
svm_model.fit(X_train, y_train)
print(svm_model.score(X_train, y_train))
print(svm_model.score(X_test, y_test))

# df = pipeline.remove_useless_col(raw_df, keep_col_list=["is_irrelevant", "input_vector"])


