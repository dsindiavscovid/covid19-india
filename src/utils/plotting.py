import json
from copy import deepcopy
from datetime import datetime, timedelta

import pandas as pd
from matplotlib import pyplot as plt, dates as mdates
from modules.data_fetcher_module import DataFetcherModule

from notebooks.nb_utils import get_observations_in_range, forecast


def plot(model_params, forecast_df, forecast_start_date, forecast_end_date, plot_name='default.png'):
    """
        Plot actual_confirmed cases vs forecasts.

        Assert that forecast_end_date is prior to the current date
        to ensure availability of actual_counts.
    """
    # Check for forecast_end_date being prior to current date
    end_date = datetime.strptime(forecast_end_date, '%m/%d/%y')
    assert end_date < datetime.now()

    # Fetch actual counts from the DataFetcher module
    data_source = model_params['data_source']
    region_name = model_params['region']
    region_type = model_params['region_type']

    # Get relevant time-series of actual counts from actual_observations
    actual_observations = get_observations_in_range(data_source, region_name, region_type,
                                                    forecast_start_date, forecast_end_date,
                                                    obs_type='confirmed')

    forecast_df['actual_confirmed'] = actual_observations

    fig, ax = plt.subplots(figsize=(15, 5))
    fig.suptitle(model_params['region'])
    ax.plot(forecast_df['index'], forecast_df['actual_confirmed'], color='blue', label="actual_confirmed")
    ax.plot(forecast_df['index'], forecast_df['confirmed_mean'], color='orange', label="predicted_confirmed")
    ax.set_ylim(ymin=0)
    ax.legend()

    plt.savefig(plot_name)


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
        ax.plot(actual['index'], actual[variable], plot_markers['observed'],
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


def plot_m1(train1_model_params, train1_run_day, train1_start_date, train1_end_date, test_run_day, test_start_date,
            test_end_date, forecast_config, uncertainty=False, rolling_average=False, plot_config='plot_config.json',
            plot_name='default.png'):
    """
        M1 plot consisting of:
            - Actuals for train1 and test intervals
            - Rolling average for train1 and test intervals
            - M1 predictions for train1 period initialized on train1 run day
            - M1 predictions for test period initialized on test run day
    """

    ## TODO: Log scale
    with open(plot_config) as fin:
        default_plot_config = json.load(fin)

    plot_config = deepcopy(default_plot_config)
    plot_config['uncertainty'] = uncertainty
    plot_config['rolling_average'] = rolling_average

    actual_start_date = (datetime.strptime(train1_start_date, "%m/%d/%y") - timedelta(days=7)).strftime("%-m/%-d/%y")

    # Get predictions
    pd_df_train = forecast(train1_model_params, train1_run_day, train1_start_date, train1_end_date, forecast_config)
    pd_df_test = forecast(train1_model_params, test_run_day, test_start_date, test_end_date, forecast_config)

    pd_df_train['index'] = pd.to_datetime(pd_df_train['index'])
    pd_df_test['index'] = pd.to_datetime(pd_df_test['index'])
    pd_df_train = pd_df_train.sort_values(by=['index'])
    pd_df_test = pd_df_test.sort_values(by=['index'])

    # Get observed data
    actual = DataFetcherModule.get_observations_for_region(
        train1_model_params['region_type'], train1_model_params['region'], train1_model_params['data_source'])
    actual = actual.set_index('observation')
    actual = actual.transpose()
    actual = actual.reset_index()
    start = actual.index[actual['index'] == actual_start_date].tolist()[0]
    end = actual.index[actual['index'] == test_end_date].tolist()[0]
    actual = actual[start : end+1]
    actual['index'] = pd.to_datetime(actual['index'])

    plot_markers = plot_config['markers']
    plot_colors = plot_config['colors']
    plot_labels = plot_config['labels']
    plot_variables = plot_config['variables']

    fig, ax = plt.subplots(figsize=(16, 12))

    for variable in plot_variables:

        # Plot observed values
        ax.plot(actual['index'], actual[variable], plot_markers['observed'],
                color = plot_colors[variable], label = plot_labels[variable]+': Observed')

        for pd_df in [pd_df_train, pd_df_test]:

            # Plot mean predictions
            if variable+'_mean' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable+'_mean'], plot_markers['predicted']['mean'],
                        color = plot_colors[variable], label = plot_labels[variable]+': Predicted')

            # Plot uncertainty in predictions
            if plot_config['uncertainty']:

                if variable+'_min' in pd_df:
                    ax.plot(pd_df['index'], pd_df[variable+'_min'], plot_markers['predicted']['min'],
                        color = plot_colors[variable], label = plot_labels[variable]+': Predicted (Min)')

                if variable+'_max' in pd_df:
                    ax.plot(pd_df['index'], pd_df[variable+'_max'], plot_markers['predicted']['max'],
                        color = plot_colors[variable], label = plot_labels[variable]+': Predicted (Max)')

            # Plot rolling average
            if plot_config['rolling_average'] == True and variable+'_ra' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable+'_ra'], plot_markers['rolling_average'],
                    color = plot_colors[variable], label = plot_labels[variable]+': Predicted (RA)')

    train_start = pd.to_datetime(train1_start_date)
    test_start = pd.to_datetime(test_start_date)

    line_height = plt.ylim()[1]
    ax.plot([train_start, train_start], [0,line_height], '--', color='brown', label='Train starts')
    ax.plot([test_start, test_start], [0,line_height], '--', color='black', label='Test starts')

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.title(train1_model_params['region'])
    plt.ylabel('No of People')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()

    plt.savefig(plot_name)


def plot_m2(train2_model_params, train2_run_day, train2_start_date, train2_end_date,
            forecast_config, uncertainty=False, rolling_average=False, plot_config='plot_config.json',
            plot_name='default.png'):
    """
        M2 plot consisting of:
            - Actuals for train2 interval and preceding weeks
            - Rolling average for train2 interval and preceding weeks
            - M2 predictions for train2 period initialized on train2 run day
    """

    ## TODO: Log scale
    with open(plot_config) as fplot:
        default_plot_config = json.load(fplot)

    plot_config = deepcopy(default_plot_config)
    plot_config['uncertainty'] = uncertainty
    plot_config['rolling_average'] = rolling_average

    actual_start_date = (datetime.strptime(train2_start_date, "%m/%d/%y") - timedelta(days=14)).strftime("%-m/%-d/%y")

    # Get predictions
    pd_df_test = forecast(train2_model_params, train2_run_day, train2_start_date, train2_end_date, forecast_config)

    pd_df_test['index'] = pd.to_datetime(pd_df_test['index'])
    pd_df = pd_df_test.sort_values(by=['index'])

    # Get observed data
    actual = DataFetcherModule.get_observations_for_region(
        train2_model_params['region_type'], train2_model_params['region'], train2_model_params['data_source'])
    actual = actual.set_index('observation')
    actual = actual.transpose()
    actual = actual.reset_index()
    start = actual.index[actual['index'] == actual_start_date].tolist()[0]
    end = actual.index[actual['index'] == train2_end_date].tolist()[0]
    actual = actual[start : end+1]
    actual['index'] = pd.to_datetime(actual['index'])

    plot_markers = plot_config['markers']
    plot_colors = plot_config['colors']
    plot_labels = plot_config['labels']
    plot_variables = plot_config['variables']

    fig, ax = plt.subplots(figsize=(16, 12))

    for variable in plot_variables:

        # Plot observed values
        ax.plot(actual['index'], actual[variable], plot_markers['observed'],
                color = plot_colors[variable], label = plot_labels[variable]+': Observed')

        # Plot mean predictions
        if variable+'_mean' in pd_df:
            ax.plot(pd_df['index'], pd_df[variable+'_mean'], plot_markers['predicted']['mean'],
                    color = plot_colors[variable], label = plot_labels[variable]+': Predicted')

        # Plot uncertainty in predictions
        if plot_config['uncertainty']:

            if variable+'_min' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable+'_min'], plot_markers['predicted']['min'],
                    color = plot_colors[variable], label = plot_labels[variable]+': Predicted (Min)')

            if variable+'_max' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable+'_max'], plot_markers['predicted']['max'],
                    color = plot_colors[variable], label = plot_labels[variable]+': Predicted (Max)')

        # Plot rolling average
        if plot_config['rolling_average'] == True and variable+'_ra' in pd_df:
            ax.plot(pd_df['index'], pd_df[variable+'_ra'], plot_markers['rolling_average'],
                color = plot_colors[variable], label = plot_labels[variable]+': Predicted (RA)')

    train2_start = pd.to_datetime(train2_start_date)

    line_height = plt.ylim()[1]
    ax.plot([train2_start, train2_start], [0,line_height], '--', color='black', label='Train starts')

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.title(train2_model_params['region'])
    plt.ylabel('No of People')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()

    plt.savefig(plot_name)


def plot_m3(train2_model_params, train1_start_date, forecast_start_date, forecast_length, forecast_config,
            rolling_average=False, uncertainty=False, plot_config='plot_config.json', plot_name='default.png'):
    """
        M3 plot consisting of:
            - Forecast from forecast_start_date
            - Actuals for preceding weeks
    """

    ## TODO: Log scale

    with open(plot_config) as fplot:
        default_plot_config = json.load(fplot)

    plot_config = deepcopy(default_plot_config)
    plot_config['uncertainty'] = uncertainty
    plot_config['rolling_average'] = rolling_average

    actual_start_date = (datetime.strptime(train1_start_date, "%m/%d/%y") - timedelta(days=14)).strftime("%-m/%-d/%y")
    forecast_run_day = (datetime.strptime(forecast_start_date, "%m/%d/%y") - timedelta(days=1)).strftime("%-m/%-d/%y")
    forecast_end_date = (datetime.strptime(forecast_start_date, "%m/%d/%y") + timedelta(days=forecast_length)).strftime("%-m/%-d/%y")

    # Get predictions
    pd_df_forecast = forecast(train2_model_params, forecast_run_day, forecast_start_date, forecast_end_date, forecast_config)

    pd_df_forecast['index'] = pd.to_datetime(pd_df_forecast['index'])
    pd_df = pd_df_forecast.sort_values(by=['index'])

    # Get observed data
    actual = DataFetcherModule.get_observations_for_region(
        train2_model_params['region_type'], train2_model_params['region'], train2_model_params['data_source'])
    actual = actual.set_index('observation')
    actual = actual.transpose()
    actual = actual.reset_index()
    start = actual.index[actual['index'] == actual_start_date].tolist()[0]
    end = actual.index[actual['index'] == forecast_run_day].tolist()[0]
    actual = actual[start : end+1]
    actual['index'] = pd.to_datetime(actual['index'])

    plot_markers = plot_config['markers']
    plot_colors = plot_config['colors']
    plot_labels = plot_config['labels']
    plot_variables = plot_config['variables']

    fig, ax = plt.subplots(figsize=(16, 12))

    for variable in plot_variables:

        # Plot observed values
        ax.plot(actual['index'], actual[variable], plot_markers['observed'],
                color = plot_colors[variable], label = plot_labels[variable]+': Observed')

        # Plot mean predictions
        if variable+'_mean' in pd_df:
            ax.plot(pd_df['index'], pd_df[variable+'_mean'], plot_markers['predicted']['mean'],
                    color = plot_colors[variable], label = plot_labels[variable]+': Predicted')

        # Plot uncertainty in predictions
        if plot_config['uncertainty']:

            if variable+'_min' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable+'_min'], plot_markers['predicted']['min'],
                    color = plot_colors[variable], label = plot_labels[variable]+': Predicted (Min)')

            if variable+'_max' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable+'_max'], plot_markers['predicted']['max'],
                    color = plot_colors[variable], label = plot_labels[variable]+': Predicted (Max)')

        # Plot rolling average
        if plot_config['rolling_average'] == True and variable+'_ra' in pd_df:
            ax.plot(pd_df['index'], pd_df[variable+'_ra'], plot_markers['rolling_average'],
                color = plot_colors[variable], label = plot_labels[variable]+': Predicted (RA)')


    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.title(train2_model_params['region'])
    plt.ylabel('No of People')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()

    plt.savefig(plot_name)