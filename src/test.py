import pandas as pd
import my_lib


if __name__ == "__main__":

    # lecture fichier
    df_data = pd.read_fwf("/data/train.txt", index_col=False, names=['annot', 'tweet'])

    df_data["label"] = df_data.apply(lambda row: row["annot"].split(",")[1], axis="columns")
    df_data["enterprise"] = df_data.apply(lambda row: row["annot"].split(",")[2][:-1], axis="columns")
    df_data.drop("annot", axis="columns", inplace=True)

    # encode to category
    le = my_lib.LabelEncoder()
    df_data["label"] = le.encode(df_data["label"], "label")
    df_data["enterprise"] = le.encode(df_data["enterprise"], "enterprise")

    # Tokenisation
    tokenizer = my_lib.processing.Tokenizer(df_data)
    df_data = tokenizer.fit()

    # model to detect irrelevant
    #model = my_lib.model.SVM().fit(X, y)

    # filter irrelevant

    # model to predict humor

    print(df_data)
