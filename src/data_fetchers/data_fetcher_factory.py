from entities.data_source import DataSource
from data_fetchers.tracker_raw import TrackerRaw
from data_fetchers.tracker_district_daily import TrackerDistrictDaily
from data_fetchers.official_data import OfficialData
from data_fetchers.rootnet_stats_history import RootnetStatsHistory


class DataFetcherFactory:

    @staticmethod
    def get_data_fetcher(data_source: DataSource):
        if data_source.__eq__(DataSource.tracker_raw_data):
            return TrackerRaw()
        elif data_source.__eq__(DataSource.tracker_district_daily):
            return TrackerDistrictDaily()
        elif data_source.__eq__(DataSource.official_data):
            return OfficialData()
        elif data_source.__eq__(DataSource.rootnet_stats_history):
            return RootnetStatsHistory()
        else:
            raise Exception("Data source is not in supported sources.")
