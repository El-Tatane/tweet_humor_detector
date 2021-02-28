import pandas as pd
import my_lib


is_test = True
nn_arch = [{"units": 256, "activation": "relu"},
           {"units": 128, "activation": "relu"},
           {"units": 4, "activation": "softmax"}]

model_input_dict = {"adaboost": {}, "elastic": {}, "rf": {}, "svm": {}, "xgboost": {"num_class": 4},
                    "nn": {"architecture": nn_arch, "num_class": 4}}


pipeline = my_lib.Pipeline()
raw_df = pipeline.load_data(is_test)

df = pipeline.encode_to_category(raw_df)
df = pipeline.tokenize(df, token_list=[])
df = pipeline.train_embedding_words(df)
X_train, X_test, y_train, y_test = pipeline.get_train_test(df, "embedding_vector", flatten=True)
pipeline.train_model(model_input_dict, X_train, y_train, X_test, y_test)