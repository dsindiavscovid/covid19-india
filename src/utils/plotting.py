import json
import os
from copy import deepcopy

import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from entities.forecast_variables import ForecastVariable
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


def plot_format(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))


def plot_vertical_lines(ax, vertical_lines):
    if vertical_lines is not None:
        for line in vertical_lines:
            ax.axvline(x=pd.to_datetime(line['date']), ls=':', color=line['color'], label=line['label'])


def plot_data(region, region_type, data_source=None, input_filepath=None, plot_config='plot_config.json',
              plot_name='default.png'):

    with open(plot_config) as fin:
        default_plot_config = json.load(fin)

    plot_config = deepcopy(default_plot_config)

    actual = DataFetcherModule.get_observations_for_region(region_type, region, data_source=data_source, smooth=False,
                                                           filepath=input_filepath)
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
        ax.plot(actual['index'], actual[variable].values, plot_markers['observed'],
                color=plot_colors[variable], label=plot_labels[variable])

    plot_format(ax)

    plt.ylabel('No of People')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()

    plt.savefig(plot_name)


def multivariate_case_count_plot(df_actual, df_smoothed=None, df_predictions_train=None, df_predictions_test=None,
                                 variables=None, column_label='Predicted mean', column_tag='mean', vertical_lines=None,
                                 title='', path=None):
    """Creates a plot of one or more variables and types of data

    Args:
        df_actual (pd.DataFrame): actual observations
        df_smoothed (pd.DataFrame, optional): smoothed observations (default: None)
        df_predictions_train (pd.DataFrame, optional): predictions for train interval (default: None)
        df_predictions_test (pd.DataFrame, optional): predictions for test interval (default: None)
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
        variables = [ForecastVariable.hospitalized.name, ForecastVariable.recovered.name,
                     ForecastVariable.deceased.name, ForecastVariable.confirmed.name]

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot actual observations, smoothed observations and predictions
    for variable in variables:
        plt.plot(df_actual['index'], df_actual[variable], 'o',
                 color=plot_colors[variable], label=f'Observed: {plot_labels[variable]}')
        if df_smoothed is not None:
            plt.plot(df_smoothed['index'], df_smoothed[variable], '-',
                     color=plot_colors[variable], label=f'Smoothed: {plot_labels[variable]}')
        for df_predictions in [df_predictions_train, df_predictions_test]:
            if df_predictions is not None:
                column = '_'.join([variable, column_tag])
                if column in df_predictions.columns:
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
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    plt.legend(legend_dict.values(), legend_dict.keys())
    plt.grid()

    if path is not None:
        plt.savefig(path)


def single_variable_case_count_plot(variable, df_actual, df_smoothed=None, df_predictions_train=None,
                                    df_predictions_test=None, column_tags=None, vertical_lines=None, title='',
                                    path=None, debug=False):
    """Creates a plot for a single variable

    Args:
        variable (str): variable to plot
        df_actual (pd.DataFrame): actual observations
        df_smoothed (pd.DataFrame, optional): smoothed observations (default: None)
        df_predictions_train (pd.DataFrame, optional): predictions for train interval (default: None)
        df_predictions_test (pd.DataFrame, optional): predictions for test interval (default: None)
        column_tags (list, optional): tags indicating column from df_predictions to be plotted (default: None)
        vertical_lines (list, optional): list of dict of vertical lines to be included in the plot of the form
            [{'date': date, 'color': color, 'label': label}]
        title (str, optional): plot title (default: '')
        path (str, optional): path to output file (default: None)
        debug (bool, optional): if True, include additional plotting (uncertainty during training)

    """

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot actual observations, smoothed observations and predictions
    plt.plot(df_actual['index'], df_actual[variable], 'o',
             color=plot_colors[variable], label=f'Observed')
    if df_smoothed is not None:
        plt.plot(df_smoothed['index'], df_smoothed[variable], '-',
                 color=plot_colors[variable], label=f'Smoothed')

    percentile_labels = []

    if debug:
        if df_predictions_train is not None and column_tags is not None:
            for tag in column_tags:
                column = '_'.join([variable, tag])
                if column in df_predictions_train.columns:
                    if tag == 'mean':
                        plt.plot(df_predictions_train['date'], df_predictions_train[column], 'x',
                                 color='black', label='Predicted mean')
                    elif tag == 'best':
                        plt.plot(df_predictions_train['date'], df_predictions_train[column], '-.',
                                 color='black', label='Predicted best fit')
                    else:
                        plt.plot(df_predictions_train['date'], df_predictions_train[column], '--',
                                 color=plot_colors[variable], label=f'Predicted percentiles')
                    percentile_labels.append(plt.text(
                        x=df_predictions_train['date'].iloc[-1],
                        y=df_predictions_train[column].iloc[-1], s=tag))
    else:
        if df_predictions_train is not None:
            plt.plot(df_predictions_train['date'], df_predictions_train[f'{variable}_mean'], 'x',
                     color='black', label='Predicted mean')

    if df_predictions_test is not None and column_tags is not None:
        for tag in column_tags:
            column = '_'.join([variable, tag])
            if column in df_predictions_test.columns:
                if tag == 'mean':
                    plt.plot(df_predictions_test['date'], df_predictions_test[column], 'x',
                             color='black', label='Predicted mean')
                elif tag == 'best':
                    plt.plot(df_predictions_test['date'], df_predictions_test[column], '-.',
                             color='black', label='Predicted best fit')
                else:
                    plt.plot(df_predictions_test['date'], df_predictions_test[column], '--',
                             color=plot_colors[variable], label=f'Predicted percentiles')
                percentile_labels.append(plt.text(
                    x=df_predictions_test['date'].iloc[-1],
                    y=df_predictions_test[column].iloc[-1], s=tag))

    adjust_text(percentile_labels, arrowprops=dict(arrowstyle="->", color='b', lw=1))

    # Plot vertical lines to mark specific events
    plot_vertical_lines(ax, vertical_lines)
    # Format plot
    plot_format(ax)

    plt.title(title)
    plt.ylabel('Case counts')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    plt.legend(legend_dict.values(), legend_dict.keys())
    plt.grid()

    if path is not None:
        plt.savefig(path)


def pdf_cdf_plot(variable, case_counts, cdf, pdf=None, case_counts_pdf=None, use_kde=True,
                 vertical_lines=None, title='', path=None):
    """Plots PDF and CDF for a variable

    Args:
        variable (str): variable to plot
        case_counts (pd.Series): variable case counts
        cdf (pd.Series): cumulative distribution function series for variable
        pdf (pd.Series, optional): probability distribution function series for variable (default: None)
        case_counts_pdf (array-like, optional): case counts to create KDE plot for PDF (default: None)
            These case counts may be modified from the original case counts to improve visualization.
        use_kde(bool, optional): use KDE plot for PDF (default: True)
        vertical_lines (list, optional): list of dict of vertical lines to be included in the plot of the form
            [{'date': date, 'color': color, 'label': label}]
        title (str, optional): plot title (default: '')
        path (str, optional): path to output file (default: None)

    """

    fig, ax_pdf = plt.subplots(figsize=(12, 8))
    ax_cdf = ax_pdf.twinx()

    # Plot PDF and CDF
    if use_kde and case_counts_pdf is not None:
        sns.kdeplot(case_counts_pdf, linestyle='dashed', color=plot_colors[variable], label=f'PDF for {variable}',
                    ax=ax_pdf)
    elif not use_kde and pdf is not None:
        ax_pdf.plot(case_counts, pdf, 'o', color=plot_colors[variable], label=f'PDF for {variable}')
    else:
        raise Exception('Must provide either case counts for KDE plot or pdf for regular plot')
    ax_cdf.plot(case_counts, cdf, '-', color=plot_colors[variable], label=f'CDF for {variable}')

    # Plot vertical lines to mark specific events
    plot_vertical_lines(ax_cdf, vertical_lines)

    ax_cdf.set_ylim(bottom=0, top=1)
    ax_pdf.set_ylim(bottom=0)
    ax_pdf.set_ylabel('PDF')
    ax_cdf.set_ylabel('CDF')
    lines, labels = ax_cdf.get_legend_handles_labels()
    lines2, labels2 = ax_pdf.get_legend_handles_labels()
    ax_pdf.legend(lines+lines2, labels+labels2)
    plt.title(title)
    ax_pdf.set_xlabel('Case counts')
    plt.grid()
    fig.tight_layout()

    if path is not None:
        plt.savefig(path)


def m1_plots(region_name, df_actual, df_smoothed, df_predictions_train, df_predictions_test,
             train1_start_date, test1_start_date, column_tags=None, variables=None, output_dir='', debug=False):
    """Creates all M1 plots

    Args:
        region_name (str): name of region
        df_actual (pd.DataFrame): actual observations
        df_smoothed (pd.DataFrame): smoothed observations
        df_predictions_train (pd.DataFrame): predictions for train interval
        df_predictions_test (pd.DataFrame): predictions for test interval
        train1_start_date (str): start date for train1 interval
        test1_start_date (str): start date for test1 interval
        column_tags (list, optional): tags indicating column from df_predictions to be plotted (default: None)
        variables (list, optional): list of variables to plot (default: None)
        output_dir (str, optional): output directory path (default: '')
        debug (bool, optional): if True, include additional plotting (uncertainty during training)

    """

    # By default, plot all variables
    if variables is None:
        variables = [ForecastVariable.hospitalized.name, ForecastVariable.recovered.name,
                     ForecastVariable.deceased.name, ForecastVariable.confirmed.name]

    # Train start and test start markers in plots
    vertical_lines = [
        {'date': train1_start_date, 'color': 'brown', 'label': 'Train starts'},
        {'date': test1_start_date, 'color': 'black', 'label': 'Test starts'}
    ]

    # Multivariate plot with M1 mean predictions
    path = os.path.join(output_dir, 'm1.png')
    multivariate_case_count_plot(df_actual, df_smoothed=df_smoothed,
                                 df_predictions_train=df_predictions_train, df_predictions_test=df_predictions_test,
                                 variables=variables, column_label='mean', column_tag='mean',
                                 vertical_lines=vertical_lines, title=f'{region_name}: M1 fit', path=path)

    # Single variable plot
    for variable in variables:
        file = f'm1_{variable}.png'
        path = os.path.join(output_dir, file)
        single_variable_case_count_plot(variable, df_actual, df_smoothed=df_smoothed,
                                        df_predictions_train=df_predictions_train,
                                        df_predictions_test=df_predictions_test, column_tags=column_tags,
                                        vertical_lines=vertical_lines, title=f'{region_name}: M1 fit - {variable}',
                                        path=path, debug=debug)


def m2_plots(region_name, df_actual, df_smoothed, df_predictions_train, train2_start_date, column_tags=None,
             variables=None, output_dir='', debug=False):
    """Creates all M2 plots

    Args:
        region_name (str): name of region
        df_actual (pd.DataFrame): actual observations
        df_smoothed (pd.DataFrame): smoothed observations
        df_predictions_train (pd.DataFrame): predictions for train interval
        train2_start_date (str): start date for train2 interval
        column_tags (list, optional): tags indicating column from df_predictions to be plotted (default: None)
        variables (list, optional): list of variables to plot (default: None)
        output_dir (str, optional): output directory path (default: '')
        debug (bool, optional): if True, include additional plotting (uncertainty during training)

    """

    # By default, plot all variables
    if variables is None:
        variables = [ForecastVariable.hospitalized.name, ForecastVariable.recovered.name,
                     ForecastVariable.deceased.name, ForecastVariable.confirmed.name]

    # Train start and test start markers in plots
    vertical_lines = [{'date': train2_start_date, 'color': 'brown', 'label': 'Train starts'}]

    # Multivariate plot with M1 mean predictions
    path = os.path.join(output_dir, 'm2.png')
    multivariate_case_count_plot(df_actual, df_smoothed=df_smoothed,
                                 df_predictions_train=df_predictions_train, df_predictions_test=None,
                                 variables=variables, column_label='mean', column_tag='mean',
                                 vertical_lines=vertical_lines, title=f'{region_name}: M2 fit', path=path)

    # Single variable plot
    for variable in variables:
        file = f'm2_{variable}.png'
        path = os.path.join(output_dir, file)
        single_variable_case_count_plot(variable, df_actual, df_smoothed=df_smoothed,
                                        df_predictions_train=df_predictions_train, df_predictions_test=None,
                                        column_tags=column_tags, vertical_lines=vertical_lines,
                                        title=f'{region_name}: M2 fit - {variable}', path=path, debug=debug)


def m2_forecast_plots(region_name, df_actual, df_smoothed, df_predictions_forecast,
                      train2_start_date, forecast_start_date, column_tags=None, variables=None, output_dir='',
                      debug=False):
    """Creates all M2 forecast plots

    Args:
        region_name (str): name of region
        df_actual (pd.DataFrame): actual observations
        df_smoothed (pd.DataFrame): smoothed observations
        df_predictions_forecast (pd.DataFrame): predictions for forecast interval
        train2_start_date (str): start date for train2 interval
        forecast_start_date (str): start date for forecast interval
        column_tags (list, optional): tags indicating column from df_predictions to be plotted (default: None)
        variables (list, optional): list of variables to plot (default: None)
        output_dir (str, optional): output directory path (default: '')
        debug (bool, optional): if True, include additional plotting (uncertainty during training)

    """

    # By default, plot all variables
    if variables is None:
        variables = [ForecastVariable.hospitalized.name, ForecastVariable.recovered.name,
                     ForecastVariable.deceased.name, ForecastVariable.confirmed.name]

    # Train start and forecast start markers in plots
    vertical_lines = [
        {'date': train2_start_date, 'color': 'brown', 'label': 'Train starts'},
        {'date': forecast_start_date, 'color': 'black', 'label': 'Forecast starts'}
    ]

    # Multivariate plot with M1 mean predictions
    path = os.path.join(output_dir, 'm2_forecast.png')
    multivariate_case_count_plot(df_actual, df_smoothed=df_smoothed,
                                 df_predictions_test=df_predictions_forecast,
                                 variables=variables, column_label='mean', column_tag='mean',
                                 vertical_lines=vertical_lines, title=f'{region_name}: M2 forecast', path=path)

    # Single variable plot
    for variable in variables:
        file = f'm2_forecast_{variable}.png'
        path = os.path.join(output_dir, file)
        single_variable_case_count_plot(variable, df_actual, df_smoothed=df_smoothed,
                                        df_predictions_test=df_predictions_forecast, column_tags=column_tags,
                                        vertical_lines=vertical_lines,
                                        title=f'{region_name}: M2 forecast - {variable}', path=path, debug=debug)


def distribution_plots(trials, variable, output_dir=''):
    """Create PDF and CDF plots

    Args:
        trials (pd.DataFrame): dataframe consisting of case counts, PDF and CDF for a variable
        variable (str): variable to plot
        output_dir (str, optional): output directory path (default: '')

    """
    trials_new = deepcopy(trials)
    num_dec = 3  # Number of decimal places to round off to
    trials_new['pdf'] = trials_new['pdf'].apply(lambda x: int(round(x, num_dec)*(10**num_dec)))
    trials_new = trials_new[trials_new['pdf'] != 0]
    trials = trials.iloc[max(0, trials_new.index.min()-1):min(trials.shape[0]-1, trials_new.index.max()+1), :]
    case_counts_pdf = []
    for i in range(trials_new.shape[0]):
        case_counts_pdf.extend([trials_new.iloc[i, :]['case_counts']]*trials_new.iloc[i, :]['pdf'])

    pdf_cdf_plot(variable, trials['case_counts'], trials['cdf'], case_counts_pdf=case_counts_pdf,
                 title=f'PDF and CDF for {variable}', path=os.path.join(output_dir, 'm2_distribution.png'))
