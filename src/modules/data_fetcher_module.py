from entities.data_source import DataSource
from data_fetchers.data_fetcher_factory import DataFetcherFactory


class DataFetcherModule(object):

    @staticmethod
    def get_observations_for_region(region_type, region_name, data_source=DataSource.tracker_district_daily,
                                    smooth=True, filepath=None):
        """Gets dataframe of case counts for a list of regions

        Args:
            region_type (str): Type of region (district or state)
            region_name (list): List of regions
            data_source (DataSource): Data source
            smooth (bool): if True, perform windowed smoothing
            filepath (str, optional): input data file path

        Returns:
            pd.DataFrame: dataframe of case counts
        """

        if not isinstance(region_name, list):
            region_name = [region_name]
        data_fetcher = DataFetcherFactory.get_data_fetcher(data_source)
        df = data_fetcher.get_observations_for_region(region_type, region_name, smooth=smooth, filepath=filepath)
        return df

    @staticmethod
    def get_regional_metadata(region_type, region_name, data_source=DataSource.tracker_district_daily,
                              filepath="../data/regional_metadata.json"):
        if not isinstance(region_name, list):
            region_name = [region_name]
        data_fetcher = DataFetcherFactory.get_data_fetcher(data_source)
        metadata = data_fetcher.get_regional_metadata(region_type, region_name, filepath)
        return metadata


if __name__ == "__main__":
    pass
