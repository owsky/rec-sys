from pandas import DataFrame, concat


def train_test_temporal_split(ratings: DataFrame, test_size: float) -> tuple[DataFrame, DataFrame]:
    """
    Splits dataset into training and test sets by respecting the time component of the ratings
    :param ratings: dataframe containing the ratings
    :param test_size: percentage of the dataset to be used for testing
    :return: tuple of train and test dataframes
    """
    train_parts = []
    test_parts = []

    for _, user_data in ratings.groupby("userId"):
        user_data = user_data.sort_values("timestamp")
        cutoff = int((1 - test_size) * len(user_data))
        train_parts.append(user_data.iloc[:cutoff])
        test_parts.append(user_data.iloc[cutoff:])

    train_df = concat(train_parts).reset_index(drop=True)
    test_df = concat(test_parts).reset_index(drop=True)

    return train_df, test_df
