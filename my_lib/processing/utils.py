from sklearn.model_selection import train_test_split


def split_data(df_data, test_size=0.2, shuffle=False, random_state=None):
    train_data, test_data = train_test_split(df_data, test_size=test_size, shuffle=shuffle, random_state=random_state)
    return train_data, test_data


def remove_useless_col(df_data, keep_col_list):
    return df_data.copy()[keep_col_list]
