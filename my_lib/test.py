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

# svm_model = my_lib.model.SVM()
# svm_model.fit(X_train, y_train)
# print(svm_model.score(X_train, y_train))
# print(svm_model.score(X_test, y_test))

# print("start train")
# rf_model = my_lib.model.HyperOptimizedRandomForest()
# rf_model.fit(X_train, y_train)
# print(rf_model.score(X_train, y_train))
# print(rf_model.score(X_test, y_test))

print("start train")
# elastic_model = my_lib.model.HyperOptimizedElasticNet()
# elastic_model.fit(X_train, y_train)
# print(elastic_model.score(X_train, y_train))
# print(elastic_model.score(X_test, y_test))

# from lazypredict.Supervised import LazyClassifier
# import numpy as np
# reg = LazyClassifier()
# model, pred = reg.fit(np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test))
# print(model)
import xgboost as xgb
import numpy as np

print(np.array(X_train).shape, np.array(y_train).shape)
print(np.array(X_test).shape, np.array(y_test).shape)

label_train = np.array(y_train)
dtrain = xgb.DMatrix(np.array(X_train), label=label_train)

label_test = np.array(y_test)
dtest = xgb.DMatrix(np.array(X_test), label=label_test)


param = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softmax', "num_class": 4}
param['nthread'] = 4
# param['eval_metric'] = 'auc'
evallist = [(dtest, 'eval'), (dtrain, 'train')]


num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist)
y_train_pred = bst.predict(dtrain)
y_test_pred = bst.predict(dtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_train, y_train_pred))
print(accuracy_score(y_test, y_test_pred))