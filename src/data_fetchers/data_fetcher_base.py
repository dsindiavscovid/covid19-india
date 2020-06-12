import abc
from typing import List

import pandas as pd

from data_fetchers.data_fetcher_utils import load_regional_metadata

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class DataFetcherBase(ABC):
    @property
    @abc.abstractmethod
    def get_observations_for_single_region(self, region_type: str, region_name: str):
        """Get a data-frame of case counts for a region
        Args:
            region_type (str): type of region
            region_name (str): name of region

        Returns:
            pd.DataFrame: data-frame of case counts for the region
        """
        pass

    def get_single_regional_metadata(self, region_type: str, region_name: str, file_path: str):

        metadata = load_regional_metadata(file_path)
        for params in metadata["regional_metadata"]:
            if params["region_type"] == region_type and params["region_name"] == region_name:
                return params["metadata"]

    def get_observations_for_region(self, region_type: str, region_name: List[str], smooth: bool = True):
        """Gets case counts for each region in the list region_name and
        creates a single data frame having case counts for the combined region

        Args:
            region_type (str):
            region_name (list[str]):
            smooth (bool):

        Returns:
            pd.DataFrame: Data frame with columns:
                region_name : combined region name
                region_type : type of region
                observation : variable observed
                    (confirmed, hospitalized, recovered, deceased and severity levels if any)
                date columns : columns of case counts for each date (from earliest to most recent date available)
        """
        df_list = []

        # Get observations for each region in region_name and concatenate them into a single data frame
        for region in region_name:
            df_region = self.get_observations_for_single_region(region_type, region)
            df_list.append(df_region)
        df = pd.concat(df_list, sort=False)

        # Get case counts for combined region by summing over individual regions
        combined_region_name = " ".join(region_name)

        if len(region_name) > 1:
            cols = [col for col in list(df) if col not in {"region_name", "region_type", "observation"}]
            df = df.groupby(["observation"])[cols].sum().reset_index()
            df.insert(0, column="region_name", value=combined_region_name)
            df.insert(1, column="region_type", value=region_type)

        # Rolling average with 3 day windows
        if smooth:
            window_size, min_window_size = 3, 1
            date_col = 3    # Beginning of date column
            df.iloc[:, date_col:] = df.iloc[:, date_col:].rolling(
                window_size, axis=1, min_periods=min_window_size).mean()
        return df

    def get_regional_metadata(self, region_type: str, region_name: List[str], file_path: str):
        """Gets metadata for a set of regions and combines it to return metadata for the combined region

        Args:
            region_type (str): type of region
            region_name (list[str]): name of region
            file_path (str): path to file containing regional metadata

        Returns:
            dict : metadata for the combined region

        Assumptions:
            It is assumed that metadata is only additive
        """
        metadata = dict()
        for region in region_name:
            regional_metadata = self.get_single_regional_metadata(region_type, region, file_path)
            for key in regional_metadata.keys():
                if key not in metadata:
                    metadata[key] = 0
                metadata[key] += regional_metadata[key]
        return metadata
