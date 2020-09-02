import pandas as pd
from data_fetchers.data_fetcher_factory import DataFetcherFactory
from data_fetchers.data_fetcher_utils import simple_data_format
from entities.data_source import DataSource


class DataFetcherModule(object):

    @staticmethod
    def get_observations_for_region(region_type, region_name, data_source=DataSource.tracker_district_daily,
                                    smooth=True, filepath=None, simple=False):
        """Gets dataframe of case counts for a list of regions

        Args:
            region_type (str): Type of region (district or state)
            region_name (list): List of regions
            data_source (DataSource): Data source
            smooth (bool): if True, perform windowed smoothing
            filepath (str, optional): input data file path
            simple (bool, optional): simplify dataframe

        Returns:
            pd.DataFrame: dataframe of case counts
        """

        if not isinstance(region_name, list):
            region_name = [region_name]
        data_fetcher = DataFetcherFactory.get_data_fetcher(data_source)
        df = data_fetcher.get_observations_for_region(region_type, region_name, smooth=smooth, filepath=filepath)
        if simple:
            df = simple_data_format(df)
        return df

    @staticmethod
    def get_regional_metadata(region_type, region_name, data_source=DataSource.tracker_data_all,
                              filepath="../data/regional_metadata.json"):
        """Gets metadata for a region

        Args:
            region_type (str): Type of region (district or state)
            region_name (list): List of regions
            data_source (DataSource): Data source
            filepath (str, optional): region metadata file path (default: "../data/regional_metadata.json")

        Returns:
            dict: region metadata
        """
        if not isinstance(region_name, list):
            region_name = [region_name]
        data_fetcher = DataFetcherFactory.get_data_fetcher(data_source)
        metadata = data_fetcher.get_regional_metadata(region_type, region_name, filepath)
        return metadata

    @staticmethod
    def get_staffing_ratios(filepath):
        return pd.read_csv(filepath)

    @staticmethod
    def get_actual_smooth_for_region(region_type, region_name, data_source, input_file_path, simple=False):
        """Get actual and smoothed data for a region

        Args:
            region_type (str): Type of region (district or state)
            region_name (list): List of regions
            data_source (DataSource): Data source
            input_file_path (str): input data file path
            simple (bool, optional): simplify dataframe

        Returns:
            dict: dictionary of dataframes with actual and smoothed observations
        """
        df_actual = DataFetcherModule.get_observations_for_region(region_type, region_name, data_source=data_source,
                                                                  smooth=False, filepath=input_file_path, simple=simple)
        df_smoothed = DataFetcherModule.get_observations_for_region(region_type, region_name, data_source=data_source,
                                                                    smooth=True, filepath=input_file_path, simple=simple)

        return {'actual': df_actual, 'smoothed': df_smoothed}

    @staticmethod
    def get_regional_data(region_type, region_name, data_source=DataSource.tracker_data_all, input_file_path=None,
                          metadata_file_path="../data/regional_metadata.json"):
        """Get actual observations and metadata for a region

        Args:
            region_type (str): Type of region (district or state)
            region_name (list): List of regions
            data_source (DataSource): Data source
            input_file_path (str): input data file path
            metadata_file_path (str, optional): region metadata file path (default: "../data/regional_metadata.json")

        Returns:
            dict: dictionary of actual observations and metadata
        """
        data = {"actual": DataFetcherModule.get_observations_for_region(region_type, region_name,
                                                                        data_source=data_source, smooth=False,
                                                                        filepath=input_file_path, simple=False),
                "metadata": DataFetcherModule.get_regional_metadata(region_type, region_name, data_source=data_source,
                                                                    filepath=metadata_file_path)}
        return data


if __name__ == "__main__":
    pass
