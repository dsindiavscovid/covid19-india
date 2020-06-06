import json
import pandas as pd
import numpy as np
from datetime import datetime

from functools import lru_cache

from entities.data_source import DataSource
from data_fetchers.data_fetcher_base import DataFetcherBase
from data_fetchers.data_fetcher_utils import get_raw_data_dict

# raw input data from rootnet api
rootnet_stats_history_url = 'https://api.rootnet.in/covid19-in/stats/history'

@lru_cache(maxsize=3)
def load_observations_data():
    data = get_raw_data_dict(rootnet_stats_history_url)
    df_result = pd.DataFrame()
    for i, _ in enumerate(data['data']):
        df_temp = pd.DataFrame.from_dict(data['data'][i]['regional'])
        df_temp = df_temp.rename(columns={'deaths':'deceased', 'discharged':'recovered'})
        df_temp['confirmed'] = df_temp['confirmedCasesForeign'] + df_temp['confirmedCasesIndian']
        df_temp['hospitalized'] = df_temp['confirmed'] - df_temp['deceased'] - df_temp['recovered']
        del df_temp['confirmedCasesForeign']
        del df_temp['confirmedCasesIndian']
        del df_temp['totalConfirmed'] # check
        df_temp.set_index('loc', inplace = True)   
        df_temp = df_temp.stack()
        df_temp = df_temp.rename_axis(['state','observation'])
        df_result = pd.concat([df_result, df_temp], axis = 1)
        date = datetime.strptime(data['data'][i]['day'], '%Y-%m-%d').strftime("%-m/%-d/%y")
        df_result = df_result.rename(columns={df_result.columns[i]:date})
    df_result.index = pd.MultiIndex.from_tuples(df_result.index, names=['region_name','observation'])
    df_result.reset_index(inplace = True)
    df_result.insert(1, column = 'region_type', value = 'state')
    df_result['region_name'] = df_result['region_name'].str.lower()
    df_result = df_result.fillna(0)
    return df_result

class RootnetStatsHistory(DataFetcherBase):

    def get_observations_for_single_region(self, region_type, region_name):
        observations_df = load_observations_data()
        region_df = observations_df[
            (observations_df["region_name"] == region_name) & (observations_df["region_type"] == region_type)]
        return region_df

