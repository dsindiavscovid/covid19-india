import pandas as pd
from functools import lru_cache

from data_fetchers.data_fetcher_base import DataFetcherBase
from data_fetchers.data_fetcher_utils import get_raw_data_dict

# Districts daily data URL from covid19india.org
data_all_url = 'https://api.covid19india.org/v4/data-all.json'
district_wise_url = 'https://api.covid19india.org/state_district_wise.json'


@lru_cache(maxsize=3)
def load_observations_data():

    # Get statecode to state dict
    data_district_wise = get_raw_data_dict(district_wise_url)
    df_statecode = pd.DataFrame.from_dict(data_district_wise)
    df_statecode = df_statecode.drop(['districtData']).T
    statecode_to_state_dict = dict(zip(df_statecode['statecode'], df_statecode.index))

    # Get raw data from URL
    data = get_raw_data_dict(data_all_url)

    for date in data.keys():
        date_dict = data[date]
        # Remove all the states which don't have district data in them
        date_dict = {state: state_dict for state, state_dict in date_dict.items() \
                     if 'districts' in state_dict.keys()}
        data[date] = date_dict

    # Remove all the dates which have 0 states with district data after pruning
    data = {date: date_dict for date, date_dict in data.items() if len(date_dict) > 0}

    # Make the districts key data the only data available for the state key
    for date in data.keys():
        for state in data[date].keys():
            # Make the districts key dict the main dict itself for a particular date, state
            data[date][state] = data[date][state]['districts']
            state_dict = data[date][state]
            # Keep only those district dicts for which cumulative data (total key) is available
            state_dict = {dist: dist_dict for dist, dist_dict in state_dict.items() \
                          if 'total' in dist_dict.keys()}
            data[date][state] = state_dict

            # Make the total key dict the main dict itself for a particular date, state, dist
            for district in data[date][state].keys():
                data[date][state][district] = data[date][state][district]['total']

            # For a particular date, state, dist, only keep those keys for which have confirmed, recovered, deceased are all available
            state_dict = {dist: dist_dict for dist, dist_dict in state_dict.items() \
                          if {'confirmed', 'recovered', 'deceased'} <= dist_dict.keys()}
            data[date][state] = state_dict

        # Remove all the states for a particular date which don't have district that satisfied above criteria
        date_dict = data[date]
        date_dict = {state: state_dict for state, state_dict in date_dict.items() if len(state_dict) > 0}
        data[date] = date_dict

    # Remove all the dates which have 0 states with district data after pruning
    data = {date: date_dict for date, date_dict in data.items() if len(date_dict) > 0}

    df_districts_all = pd.DataFrame(columns=['date', 'state', 'district', 'confirmed', 'active',
                                             'recovered', 'deceased', 'tested', 'migrated'])
    for date in data.keys():
        for state in data[date].keys():
            df_date_state = pd.DataFrame.from_dict(data[date][state]).T.reset_index()
            df_date_state = df_date_state.rename({'index': 'district'}, axis='columns')
            df_date_state['active'] = df_date_state['confirmed'] - \
                                      (df_date_state['recovered'] + df_date_state['deceased'])
            df_date_state['state'] = statecode_to_state_dict[state]
            df_date_state['date'] = date
            df_districts_all = pd.concat([df_districts_all, df_date_state], ignore_index=True)

    numeric_cols = ['confirmed', 'active', 'recovered', 'deceased', 'tested', 'migrated']
    df_districts_all.loc[:, numeric_cols] = df_districts_all.loc[:, numeric_cols].apply(pd.to_numeric)
    df_districts_all['date'] = pd.to_datetime(df_districts_all['date'], format = "%Y-%m-%d")
    df_districts_all['date'] = df_districts_all['date'].dt.strftime("%-m/%-d/%y")
    df_districts_all.set_index('date',  inplace = True)
    return df_districts_all


class TrackerDataAll(DataFetcherBase):

    def get_observations_for_single_region(self, region_type, region_name, filepath=None):
        region_name = region_name.capitalize()
        observations_df = load_observations_data()
        region_df = observations_df[
            (observations_df["district"] == region_name) & (region_type.lower() == 'district')]
        region_df = region_df.rename(columns = {'active': 'hospitalized'})
        region_df.index.name = 'index'
        region_df = region_df.drop(['state', 'district'], axis = 1)
        region_df = region_df.transpose()
        region_df = region_df.reset_index()
        region_df = region_df.rename(columns = {'index' : "observation"})
        region_df.insert(0, 'region_name', region_name.lower())
        region_df.insert(1, 'region_type', region_type.lower())
        return region_df
