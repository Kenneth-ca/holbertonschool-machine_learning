#!/usr/bin/env python3
"""
Script that preprocess bitcoin data to forecast
"""
import pandas as pd


def cleaning(filename):
    """
    Performs data cleaning to a dataframe read from a csv file
    :param filename: path to the file
    :return: a cleaned dataframe
    """
    # Read the file into a pandas dataframe
    with open(filename, "r") as f:
        df = pd.read_csv(f, delimiter=',')
    # Convert the timestamp into a datetime format
    df["Date"] = pd.to_datetime(df['Timestamp'], unit='s')
    # Drop duplicates
    df = df.drop_duplicates(subset="Timestamp")
    # Take relevant data as before 2017 does not add to the model
    df = df[df["Date"] >= "2017"]
    # Resample by hours
    df = df.set_index("Date")
    df = df.resample("H").agg({"Open": "mean", "High": "mean",
                               "Low": "mean", "Close": "mean",
                               "Volume_(BTC)": "sum",
                               "Volume_(Currency)": "sum",
                               "Weighted_Price": "mean"})
    df.reset_index(inplace=True)
    # Drop columns that doesn't help predict the model
    df.pop("Volume_(BTC)")
    df.pop("Volume_(Currency)")
    df.pop("High")
    df.pop("Low")
    # Fill NaN
    df.ffill(inplace=True)
    print(df.head())
    return df


def preprocess(df):
    """
    Preprocess a dataframe for a forecasting
    :param df:
    :return:
    """
    date_time = pd.to_datetime(df.pop('Date'), format='%d.%m.%Y %H:%M:%S')

    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df


def save_csv(filename, dataframe):
    """
    Auxiliar function to save a dataframe into a csv file
    :param filename: path to save the file
    :param dataframe: dataframe to be saved
    :return:
    """
    dataframe.to_csv(filename, index=False)
