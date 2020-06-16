import pandas as pd

from pathlib import Path
from functools import lru_cache

from data_fetchers.data_fetcher_base import DataFetcherBase

# TODO: 1. Add file_path to config 2. Choose one file format or add support for multiple formats in config

file_path_dict = {
    "mumbai": {
        "format": 2,
        "path": "../../data/mumbai_data.csv"
    },
    "pune": {
        "format": 1,
        "path": "../../data/pune_data.csv"
    }
}


@lru_cache(maxsize=3)
def load_observations_data(region_type, region_name):
    """Load case counts from CSV depending on file format

    Args:
        region_type (str): type of region
        region_name (str): name of region

    Returns:
        pd.Dataframe: daily cumulative case counts for region
    """
    d = file_path_dict[region_name]
    file_path = Path(__file__).parent / d["path"]
    file_format = d["format"]
    if file_format == 1:
        return load_observations_data_format_1(region_type, region_name, file_path)
    elif file_format == 2:
        return load_observations_data_format_2(region_type, region_name, file_path)
    else:
        raise Exception("File format not supported")


def load_observations_data_format_1(region_type, region_name, file_path):
    """Load case counts from CSV
    CSV has columns:
            date, total_infected, active, recovered, deceased, (additional unnamed column)

    Args:
        region_type (str): type of region
        region_name (str): name of region
        file_path (str): path to file

    Returns:
        pd.Dataframe: daily cumulative case counts for region
    """
    df = pd.read_csv(
        file_path, usecols=['date', 'total_infected', 'active', 'recovered', 'deceased'], index_col='date', header=0)
    df = df.rename(columns={'total_infected': 'confirmed', 'active': 'hospitalized'})
    df.index = pd.to_datetime(df.index).strftime('%-m/%-d/%y')
    df = df.transpose().reset_index().rename(columns={'index': 'observation'}).rename_axis(None)
    df.insert(0, column='region_name', value=region_name)
    df.insert(1, column='region_type', value=region_type)
    return df


def load_observations_data_format_2(region_type, region_name, file_path):
    """Load case counts from CSV
        CSV has columns:
            State, District, Ward/block name, Ward number (if applicable), Date, Total cases, Active cases,
            Mild cases (isolated), Moderate cases (hospitalized), Severe cases (In ICU),
            Critical cases (ventilated patients), Recovered cases, Fatalities

        Args:
            region_type (str): type of region
            region_name (str): name of region
            file_path (str): path to file

        Returns:
            pd.Dataframe: daily cumulative case counts for region
        """
    df = pd.read_csv(file_path, header=0)
    df = df.drop(range(3))
    del df['State']
    del df['District']
    del df['Ward/block name']
    del df['Ward number (if applicable)']
    del df['Mild cases (isolated)']
    del df['Moderate cases (hospitalized)']
    del df['Severe cases (In ICU)']
    del df['Critical cases (ventilated patients)']
    df.columns = ['date', 'confirmed', 'hospitalized', 'recovered', 'deceased']
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index).strftime('%-m/%-d/%y')
    df = df.transpose().reset_index().rename(columns={'index': 'observation'}).rename_axis(None)
    df.insert(0, column='region_name', value=region_name)
    df.insert(1, column='region_type', value=region_type)
    df = df.dropna(subset=['date'], how='all')
    return df


class DirectCSV(DataFetcherBase):

    def get_observations_for_single_region(self, region_type, region_name):
        observations_df = load_observations_data(region_type, region_name)
        region_df = observations_df[
            (observations_df["region_name"] == region_name) & (observations_df["region_type"] == region_type)]
        return region_df

