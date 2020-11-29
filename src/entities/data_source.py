import enum


@enum.unique
class DataSource(str, enum.Enum):
    tracker_raw_data = "tracker_raw_data"
    tracker_district_daily = "tracker_district_daily"
    direct_csv = "direct_csv"
    official_data = "official_data"
    rootnet_stats_history = "rootnet_stats_history"
    tracker_data_all = "tracker_data_all"
