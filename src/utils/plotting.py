import json
from copy import deepcopy
from datetime import datetime, timedelta

import pandas as pd
from matplotlib import pyplot as plt, dates as mdates
from modules.data_fetcher_module import DataFetcherModule

# TODO: Should this be in a plot config json

plot_colors = {
    "confirmed": "C0",
    "recovered": "green",
    "hospitalized": "orange",
    "deceased": "red"
}

plot_labels = {
    "confirmed": "Confirmed cases",
    "recovered": "Recovered cases",
    "hospitalized": "Hospitalized cases",
    "deceased": "Deceased cases"
}

all_variables = ['hospitalised', 'deceased', 'recovered', 'confirmed']


def plot_format(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))


def plot_vertical_lines(ax, vertical_lines):
    if vertical_lines is not None:
        for line in vertical_lines:
            ax.axvline(x=line['date'], ls=':', color=line['color'], label=line['label'])


def get_observations_for_plot(region_type, region_name, start_date, end_date, data_source, smooth=False, filepath=None):

    df_actual = DataFetcherModule.get_observations_for_region(region_type, region_name,
                                                              data_source=data_source, smooth=smooth, filepath=filepath)
    df_actual = df_actual.set_index('observation')
    df_actual = df_actual.transpose().reset_index()
    start = df_actual.index[df_actual['date'] == start_date].tolist()[0]
    end = df_actual.index[df_actual['date'] == end_date].tolist()[0]
    df_actual = df_actual[start: end + 1]
    df_actual['date'] = pd.to_datetime(df_actual['date'])

    return df_actual


def plot_data(region, region_type, data_source=None, plot_config='plot_config.json', plot_name='default.png'):

    with open(plot_config) as fin:
        default_plot_config = json.load(fin)

    plot_config = deepcopy(default_plot_config)

    actual = DataFetcherModule.get_observations_for_region(region_type, region, data_source=data_source, smooth=False)
    actual.drop(columns=['region_name', 'region_type'], inplace=True)
    actual = actual.set_index('observation').transpose().reset_index()
    actual['index'] = pd.to_datetime(actual['index'])
    actual = actual.loc[~ (actual.select_dtypes(include=['number']) == 0).all(axis='columns'), :]

    plot_markers = plot_config['markers']
    plot_colors = plot_config['colors']
    plot_labels = plot_config['labels']
    plot_variables = plot_config['variables']

    fig, ax = plt.subplots(figsize=(16, 12))

    for variable in plot_variables:
        plt.plot(actual['index'], actual[variable], plot_markers['observed'],
                 color=plot_colors[variable], label=plot_labels[variable])

    plot_format(ax)

    plt.ylabel('No of People')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()

    plt.savefig(plot_name)


def multivariate_case_count_plot(df_actual, df_smoothed=None, df_predictions=None, variables=None,
                                 column_label='Predictions', column_tag='', vertical_lines=None, title='', path=None):
    """Creates a plot of one or more variables and types of data

    Args:
        df_actual (pd.DataFrame): actual observations
        df_smoothed (pd.DataFrame, optional): smoothed observations (default: None)
        df_predictions (pd.DataFrame, optional): predictions (default: None)
        variables (list, optional): list of variables to plot (default: None)
        column_label (str, optional): label for type of column such as mean, percentile (default: 'Predictions')
        column_tag (str, optional): tag indicating column from df_predictions to be plotted (default: '')
        vertical_lines (list, optional): list of dict of vertical lines to be included in the plot of the form
            [{'date': date, 'color': color, 'label': label}] (default: None)
        title (str, optional): plot title (default: '')
        path (str, optional): path to output file (default: None)

    """

    # By default, plot all variables
    if variables is None:
        variables = all_variables

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot actual observations, smoothed observations and predictions
    for variable in variables:
        plt.plot(df_actual['date'], df_actual[variable], '-o',
                 color=plot_colors[variable], label=f'Observed: {plot_labels[variable]}')
        if df_smoothed is not None:
            plt.plot(df_smoothed['date'], df_smoothed[variable], '-',
                     color=plot_colors[variable], label=f'Smoothed: {plot_labels[variable]}')
        if df_predictions is not None:
            column = '_'.join([variable, column_tag])
            plt.plot(df_predictions['date'], df_predictions[column], '--',
                     color=plot_colors[variable], label=f'{column_label}: {plot_labels[variable]}')

    # Plot vertical lines to mark specific events
    plot_vertical_lines(ax, vertical_lines)
    # Format plot
    plot_format(ax)

    plt.title(title)
    plt.ylabel('Case counts')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()

    if path is not None:
        plt.savefig(path)


def single_variable_case_count_plot(variable, df_actual, df_smoothed=None, df_predictions=None, column_tags=None,
                                    vertical_lines=None, title='', path=''):
    """Creates a plot for a single variable

    Args:
        variable (str): variable to plot
        df_actual (pd.DataFrame): actual observations
        df_smoothed (pd.DataFrame, optional): smoothed observations (default: None)
        df_predictions (pd.DataFrame, optional): predictions (default: None)
        column_tags (list, optional): tags indicating column from df_predictions to be plotted (default: None)
        vertical_lines (list, optional): list of dict of vertical lines to be included in the plot of the form
            [{'date': date, 'color': color, 'label': label}]
        title (str, optional): plot title (default: '')
        path (str, optional): path to output file (default: None)

    """

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot actual observations, smoothed observations and predictions
    plt.plot(df_actual['date'], df_actual[variable], '-o',
             color=plot_colors[variable], label=f'Observed: {plot_labels[variable]}')
    if df_smoothed is not None:
        plt.plot(df_smoothed['date'], df_smoothed[variable], '-',
                 color=plot_colors[variable], label=f'Smoothed: {plot_labels[variable]}')
    if df_predictions is not None and column_tags is not None:
        for tag in column_tags:
            column = '_'.join([variable, tag])
            if tag == 'mean':
                plt.plot(df_predictions['date'], df_predictions[column], 'x',
                         color=plot_colors[variable], label=f'Predicted mean: {plot_labels[variable]}')
            else:
                plt.plot(df_predictions['date'], df_predictions[column], '--',
                         color=plot_colors[variable], label=f'Predicted percentile: {plot_labels[variable]}')
            # Label the percentile curves

    # Plot vertical lines to mark specific events
    plot_vertical_lines(ax, vertical_lines)
    # Format plot
    plot_format(ax)

    plt.title(title)
    plt.ylabel('Case counts')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()

    if path is not None:
        plt.savefig(path)


def m1_plots(df_predictions, train1_model_params,
             train1_run_day, train1_start_date, train1_end_date,
             test_run_day, test_start_date, test_end_date, forecast_config):

    # TODO: Do we fetch actuals and get forecasts here or outside

    # Start and end dates of plots
    plot_start_date = (datetime.strptime(train1_start_date, "%m/%d/%y") - timedelta(days=7)).strftime("%-m/%-d/%y")
    plot_end_date = test_end_date

    # Get actual observations and smoothed data
    df_actual = get_observations_for_plot(train1_model_params['region_type'], train1_model_params['region'],
                                          plot_start_date, plot_end_date,
                                          train1_model_params['data_source'], smooth=False,
                                          filepath=train1_model_params['input_filepath'])
    df_smoothed = get_observations_for_plot(train1_model_params['region_type'], train1_model_params['region'],
                                            plot_start_date, plot_end_date,
                                            train1_model_params['data_source'], smooth=True,
                                            filepath=train1_model_params['input_filepath'])

    # Train start and test start markers in plots
    vertical_lines = [
        {'date': train1_start_date, 'color': 'brown', 'label': 'Train starts'},
        {'date': test_start_date, 'color': 'black', 'label': 'Test starts'}
    ]

    region_name = " ".join(train1_model_params['region_name'])

    # Multivariate plot with M1 mean predictions
    multivariate_case_count_plot(df_actual, df_smoothed=df_smoothed, df_predictions=df_predictions,
                                 column_label='mean', column_tag='mean', vertical_lines=vertical_lines,
                                 title=f'{region_name}: M1 fit')

    # Get percentiles to plot
    uncertainty_params = forecast_config['model_parameters']['uncertainty_parameters']
    percentiles = uncertainty_params['percentiles'] + uncertainty_params['ci']
    percentiles = [str(i) for i in percentiles]

    # Single variable plot
    for variable in all_variables:
        single_variable_case_count_plot(variable, df_actual, df_smoothed=df_smoothed, df_predictions=df_predictions,
                                        column_tags=percentiles, vertical_lines=vertical_lines,
                                        title=f'{region_name}: M1 fit - {variable}')
