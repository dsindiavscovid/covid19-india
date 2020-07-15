import pandas as pd

from pathlib import Path
from functools import lru_cache

from data_fetchers.data_fetcher_base import DataFetcherBase


@lru_cache(maxsize=3)
def load_observations_data(region_type, region_name, filepath):
    """Load case counts from CSV
    CSV has columns:
            date, total_infected, active, recovered, deceased, (additional unnamed column)

    Args:
        region_type (str): type of region
        region_name (str): name of region
        filepath (str): path to file

    Returns:
        pd.Dataframe: daily cumulative case counts for region
    """
    df = pd.read_csv(
        filepath, usecols=['date', 'total_infected', 'active', 'recovered', 'deceased'], index_col='date', header=0)
    df = df.rename(columns={'total_infected': 'confirmed', 'active': 'hospitalized'})
    df.index = pd.to_datetime(df.index).strftime('%-m/%-d/%y')
    df = df.transpose().reset_index().rename(columns={'index': 'observation'}).rename_axis(None)
    df.insert(0, column='region_name', value=region_name)
    df.insert(1, column='region_type', value=region_type)
    print(df)
    return df


class DirectCSV(DataFetcherBase):

    def get_observations_for_single_region(self, region_type, region_name, filepath=None):
        if filepath is None or filepath == "":
            raise Exception("Input file path not provided")
        observations_df = load_observations_data(region_type, region_name, filepath)
        region_df = observations_df[
            (observations_df["region_name"] == region_name) & (observations_df["region_type"] == region_type)]
        return region_df

