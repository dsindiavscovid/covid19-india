import json
import urllib.request

import pandas as pd


def get_raw_data_dict(input_url):
    """Gets dictionary of raw data from a given URL"""
    with urllib.request.urlopen(input_url) as url:
        data_dict = json.loads(url.read().decode())
        return data_dict


def load_regional_metadata(file_path):
    """Gets dictionary of regional metadata from a given file path"""
    with open(file_path, 'r') as fp:
        return json.load(fp)


def smooth_data(df, window_size=3):
    min_window_size = 1
    date_col = 3  # Beginning of date column
    df.iloc[:, date_col:] = df.iloc[:, date_col:].rolling(
        window_size, axis=1, center=True, min_periods=min_window_size).mean()
    return df


def simple_data_format(df):
    df.drop(columns=['region_name', 'region_type'], inplace=True)
    df = df.set_index('observation').transpose().reset_index()
    df['index'] = pd.to_datetime(df['index'])
    df = df.loc[~ (df.select_dtypes(include=['number']) == 0).all(axis='columns'), :]
    return df
