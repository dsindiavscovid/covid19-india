from functools import lru_cache

import pandas as pd
from data_fetchers.data_fetcher_base import DataFetcherBase

# v4 data all URL from covid19india.org
data_url = 'https://api.covid19india.org/csv/latest/districts.csv'


@lru_cache(maxsize=3)
def load_observations_data():
    df = pd.read_csv(data_url)
    df.columns = [x.lower() for x in df.columns]
    df['active'] = df['confirmed'] - (df['recovered'] + df['deceased'])
    numeric_cols = ['confirmed', 'active', 'recovered', 'deceased', 'tested', 'other']
    df.loc[:, numeric_cols] = df.loc[:, numeric_cols].apply(pd.to_numeric)
    df = df.fillna(0)
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df['date'] = [d.strftime("%-m/%-d/%y") for d in df['date']]
    return df


class TrackerDistricts(DataFetcherBase):

    def get_observations_for_single_region(self, region_type, region_name, filepath=None):
        region_name = " ".join([word.capitalize() for word in region_name.split(" ")])
        observations_df = load_observations_data()
        region_df = observations_df[
            (observations_df["district"] == region_name) & (region_type.lower() == 'district')]
        region_df = region_df.rename(columns={'active': 'hospitalized'})
        region_df = region_df.drop(['state', 'district'], axis=1)
        region_df.set_index('date', inplace=True)
        region_df = region_df.transpose()
        region_df = region_df.reset_index()
        region_df = region_df.rename(columns={'index': "observation"})
        region_df.insert(0, 'region_name', region_name.lower())
        region_df.insert(1, 'region_type', region_type.lower())
        return region_df
