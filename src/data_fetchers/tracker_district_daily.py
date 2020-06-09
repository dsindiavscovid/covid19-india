from datetime import datetime
from functools import lru_cache

import pandas as pd

from data_fetchers.data_fetcher_base import DataFetcherBase
from data_fetchers.data_fetcher_utils import get_raw_data_dict

# Districts daily data URL from covid19india.org
district_daily_url = 'https://api.covid19india.org/districts_daily.json'


@lru_cache(maxsize=3)
def load_observations_data():
    """Returns data frame of district-wise case counts from covid19india.org districts_daily API

    Returns:
        pd.DataFrame: district wise daily cumulative case counts
    """

    # Get raw data from URL
    raw_data = get_raw_data_dict(district_daily_url)["districtsDaily"]

    # Create empty data frame
    dates = pd.date_range(start="2020-04-01", end=datetime.today()).strftime("%Y-%m-%d")
    columns = ["region_name", "observation"]
    columns.extend(dates)
    df = pd.DataFrame(columns=columns)

    # Loop over dictionary of raw data and append district-wise data to data frame
    for state_ut in raw_data:
        for district in raw_data[state_ut]:
            df_temp = pd.DataFrame(raw_data[state_ut][district])
            df_temp = df_temp.drop('notes', axis=1)
            df_temp = df_temp.set_index('date').transpose().reset_index()
            df_temp = df_temp.rename(columns={'index': "observation"}).rename_axis(None)
            df_temp.insert(0, column="region_name", value=district.lower().replace(',', ''))
            df = pd.concat([df, df_temp], axis=0, ignore_index=True, sort=False)

    # Convert data frame to required format
    df.insert(1, column="region_type", value="district")
    df = df.replace("active", "hospitalized")
    df = df.sort_values(by=["observation"])
    df = df.fillna(0)
    dates = pd.date_range(start="4/1/20", end=datetime.today()).strftime("%-m/%-d/%y")
    new_columns = ["region_name", "region_type", "observation"]
    new_columns.extend(dates)
    df = df.rename(columns=dict(zip(df.columns, new_columns)))

    return df


class TrackerDistrictDaily(DataFetcherBase):

    def get_observations_for_single_region(self, region_type, region_name):
        observations_df = load_observations_data()
        region_df = observations_df[
            (observations_df["region_name"] == region_name) & (observations_df["region_type"] == region_type)]
        return region_df
