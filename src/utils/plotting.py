import json
from copy import deepcopy

import pandas as pd
from matplotlib import pyplot as plt, dates as mdates
from modules.data_fetcher_module import DataFetcherModule



def plot_data(region, region_type, data_source=None, plot_config='plot_config.json', plot_name='default.png'):

    with open(plot_config) as fin:
        default_plot_config = json.load(fin)

    plot_config = deepcopy(default_plot_config)

    actual = DataFetcherModule.get_observations_for_region(region_type, region, data_source=data_source, smooth=False)
    actual.drop(columns=['region_name', 'region_type'], inplace=True)
    actual = actual.set_index('observation').transpose().reset_index()
    actual['index'] = pd.to_datetime(actual['index'])
    actual = actual.loc[~ (actual.select_dtypes(include=['number']) == 0).all(axis='columns'), :] # CHECK THIS

    plot_markers = plot_config['markers']
    plot_colors = plot_config['colors']
    plot_labels = plot_config['labels']
    plot_variables = plot_config['variables']

    fig, ax = plt.subplots(figsize=(16, 12))

    for variable in plot_variables:
        plt.plot(actual['index'], actual[variable], plot_markers['observed'],
                 color=plot_colors[variable], label=plot_labels[variable])

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.ylabel('No of People')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()

    plt.savefig(plot_name)
