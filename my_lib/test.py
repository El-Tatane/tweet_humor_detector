import my_lib

pipeline = my_lib.Pipeline()
raw_df = pipeline.load_data()

# DEBUG
raw_df = raw_df[0:10]

raw_df = pipeline.encode_to_category(raw_df)
raw_df = pipeline.tokenize(raw_df)
raw_df.head()



df = pipeline.remove_useless_col(raw_df, keep_col_list=["is_irrelevant", "number_vector"])

for idx in range(df.shape[0]):
    while len(df.loc[idx, "number_vector"]) < 280:
        df.loc[idx, "number_vector"].append(-1)

train_data, test_data = pipeline.split_data(df)
print(f"SHAPE -> train : {train_data.shape}, test : {test_data.shape}")

X_train = train_data["number_vector"].values
y_train = train_data["is_irrelevant"].values
X_test = test_data["number_vector"].values
y_test = test_data["is_irrelevant"].values

svm_model = my_lib.model.SVM()
svm_model.fit(X_train, y_train)
print(svm_model.score(X_train, y_train))
print(svm_model.score(X_test, y_train))




