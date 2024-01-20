""" Data Frame Operations utility file for file reading, data frame generation, and data frame filtering"""
import pandas as pd
from sklearn.model_selection import train_test_split


def get_df_from_csv_file(path, columns=None, seperator=","):
    """
    Reads a csv file and turns it into data frame

    :param path: (str) path to the csv file
    :param columns: (str[]) (default is None) columns specified to get from the csv file
    :param seperator: (str) (default is ",") seperator used in the csv file

    :returns data_df: (pandas.DataFrame)
    """

    if columns is None:
        data_df = pd.read_csv(path, sep=seperator)
    else:
        data_df = pd.read_csv(path, usecols=columns, sep=seperator)

    return data_df


def filter_csv_data(df, text_col_name, label_col_name):
    """
    Filters data and returns a final df to be used in model training

    :param df: (Pandas.DataFrame) data frame to be filtered
    :param text_col_name: (str) text columns name in df
    :param label_col_name: (str) label columns name in df

    :returns df_final:
    """

    # Filter empty columns
    df_filtered = df.dropna()

    # Lower-case data for standardization
    df_filtered = df_filtered.applymap(lambda x: str(x).lower())

    # Create a final data frame
    df_final = pd.DataFrame(columns=["Summary", "Category"])
    df_final["Summary"] = df_filtered[text_col_name]
    df_final["Category"] = df_filtered[label_col_name]

    print(df_final.head())
    return df_final


def split_dataset_properly(df, split_size=0.30, truncate=False):
    """
    Splits dataset properly by even distribution of errors
    :author Barış Giray Akman


    :param df: (Pandas.DataFrame)

    :returns x_train, x_test, y_train, y_test
    """

    df.drop_duplicates()
    error_types = df["Category"].unique()

    x_train, x_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for i in range(len(error_types)):
        # Get unique error classifications and their corresponding descriptions
        rows = df[df["Category"] == error_types[i]]
        rows = rows.sample(frac=1)

        # Create sub data frames
        x_train_sub, x_test_sub, y_train_sub, y_test_sub = train_test_split(rows["Summary"], rows["Category"],
                                                                            test_size=split_size)

        # Concat data frames for each unique classification
        if truncate:
            x_test = pd.concat((x_test, pd.DataFrame(x_test_sub)[:0]), axis=0)
            y_test = pd.concat((y_test, pd.DataFrame(y_test_sub)[:0]), axis=0)
            x_train = pd.concat((x_train, pd.DataFrame(x_train_sub)[:150]), axis=0)
            y_train = pd.concat((y_train, pd.DataFrame(y_train_sub)[:150]), axis=0)
        else:
            x_test = pd.concat((x_test, x_test_sub), axis=0)
            y_test = pd.concat((y_test, y_test_sub), axis=0)
            x_train = pd.concat((x_train, x_train_sub), axis=0)
            y_train = pd.concat((y_train, y_train_sub), axis=0)

    x_train, x_test, y_train, y_test = x_train.squeeze(), x_test.squeeze(), y_train.squeeze(), y_test.squeeze()

    return x_train, x_test, y_train, y_test
