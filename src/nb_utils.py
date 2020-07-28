import json
import os

from copy import deepcopy
from datetime import datetime, timedelta

import mlflow
import pandas as pd
from configs.base_config import ForecastingModuleConfig
from configs.base_config import ModelEvaluatorConfig
from configs.base_config import TrainingModuleConfig
from matplotlib import pyplot as plt, dates as mdates
from model_wrappers.model_factory import ModelFactory
from modules.data_fetcher_module import DataFetcherModule
from modules.forecasting_module import ForecastingModule
from modules.model_evaluator import ModelEvaluator
from modules.training_module import TrainingModule
from utils.plotting import m1_plots, m2_plots, m2_forecast_plots, distribution_plots
from utils.data_transformer_helper import loss_json_to_dataframe


def parse_params(parameters, interval='Train1'):
    """
        Flatten the params dictionary to enable logging
    of the parameters.

    Assumptions:
        There is a maximum of one level of nesting.
        Ensured using an assert statement for now.

    Sample_input:
        {
            'LatentEbyCRatio': {
                '4/7/20': 0.5648337712691847,
                '4/17/20': 1.1427545912005197
            },
            'LatentIbyCRatio': {
                '4/7/20': 0.9610881623714099,
                '4/17/20': 0.6742970940209254
            }
        }

    Output:
        {
            'Train1_LatentEbyCRatio_4/7/20': 0.5648337712691847,
            'Train1_LatentEbyCRatio_4/17/20': 1.1427545912005197,
            'Train1_LatentIbyCRatio_4/7/20': 0.9610881623714099,
            'Train1_LatentIbyCRatio_4/17/20': 0.6742970940209254
        }
    """
    param_dict = dict()  # The flattened dictionary to return
    for param in parameters:
        if isinstance(parameters[param], dict):
            for key in parameters[param]:
                assert (not isinstance(parameters[param][key], dict))

                param_dict[interval + '_' + param + '_' + key] = parameters[param][key]
        else:
            param_dict[interval + '_' + param] = parameters[param]
    return param_dict


def parse_metrics(metrics, interval='Train1'):
    """
        Flatten the list of loss metrics to enable logging.
    
    Note:
        Consider only losses computed on individual buckets.
    """
    metric_dict = dict()
    for metric in metrics:
        prefix = interval + '_' + metric['metric_name'] + '_'
        variables = metric['variable_weights']
        if len(variables) == 1:
            metric_dict[prefix + variables[0]['variable']] = metric['value']
            metric_dict[prefix + variables[0]['variable'] + '_weight'] = variables[0]['weight']
    return metric_dict


def train_eval(region, region_type, 
               train1_start_date, train1_end_date, 
               train2_start_date, train2_end_date, run_day,
               test_start_date, test_end_date,
               default_train_config, default_test_config,
               max_evals=1000, data_source=None, input_filepath=None,
               mlflow_log=True, name_prefix=None):
    """
        Run train and evaluation for (basic) SEIR model.
    
    Arguments:
        region, region_type : Region info corresponding to the run
        train1_start_date, train1_end_date : Train1 durations
        train2_start_date, train2_end_date : Train2 durations
        test_start_date, test_end_date, run_day : Test durations
        default_train_config : Default train config (loaded from train_config.json)
        default_test_config : Default test config (loaded from test_config.json)
        max_evals : number of search evaluations for SEIR (default: 1000)
        data_source : Data source for picking the region data
        mlflow_log : Experiment logged using MLFlow (default: True)
        name_prefix : In case of non-MLFlow experiment, string prefix to
                      enable easy indexing of experiments

    Note:
        date_format : %-m/%-d/%-y

    Returns: 
        params : Run parameters to be logged
        metrics : Metrics collected from the run 
    
    Output files saved : (name_prefix added in the case of non-MLflow experiments)
        Train1 : train1_output.json (name_prefix + '_train1_output.json')
        Train2 : train2_output.json (name_prefix + '_train2_output.json')
        Test   : test1_output.json   (name_prefix + '_test1_output.json')
    """
    
    # Save metrics and params for logging
    
    params = dict()
    metrics = dict()

    params['region'] = " ".join(region)
    params['region_type'] = region_type
    params['train1_start_date'] = train1_start_date
    params['train1_end_date'] = train1_end_date
    params['train2_start_date'] = train2_start_date
    params['train2_end_date'] = train2_end_date
    params['run_day'] = run_day
    params['test_start_date'] = test_start_date
    params['test_end_date'] = test_end_date
    params['data_source'] = data_source

    train_config = deepcopy(default_train_config)
    train_config['data_source'] = data_source
    train_config['region_name'] = region
    train_config['region_type'] = region_type
    train_config['train_start_date'] = train1_start_date
    train_config['train_end_date'] = train1_end_date
    train_config['search_parameters']['max_evals'] = max_evals
    train_config['input_filepath'] = input_filepath

    # model parameters
    model_params = dict()
    model_params['data_source'] = data_source
    model_params['region'] = region
    model_params['region_type'] = region_type
    model_params['model_type'] = train_config['model_class']
    model_params['input_filepath'] = input_filepath

    train1_model_params = deepcopy(model_params)
    train2_model_params = deepcopy(model_params)

    if mlflow_log:
        train_config['output_filepath'] = 'train1_output.json'
    else:
        assert name_prefix is not None
        train_config['output_filepath'] = name_prefix + '_train1_output.json'

    train_module_config = TrainingModuleConfig.parse_obj(train_config)
    train_results = TrainingModule.from_config(train_module_config)

    train1MAPE = 0
    train1RMSLE = 0
    for metric in train_results['train_metric_results']:
        if metric['metric_name'] == 'mape':
            train1MAPE += metric['value']
        if metric['metric_name'] == 'rmsle':
            train1RMSLE += metric['value']

    metrics['Train1RMLSE'] = train1RMSLE
    metrics['Train1MAPE'] = train1MAPE 
    metrics.update(parse_metrics(train_results['train_metric_results'], 'Train1'))
    train1_model_params['model_parameters'] = train_results['model_parameters']

    test_config = deepcopy(default_test_config)
    test_config['data_source'] = data_source
    test_config['region_name'] = region
    test_config['region_type'] = region_type
    test_config['test_start_date'] = test_start_date
    test_config['test_end_date'] = test_end_date
    test_config['run_day'] = run_day
    test_config['model_parameters'].update(train_results['model_parameters'])
    test_config['input_filepath'] = input_filepath
        
    if mlflow_log:
        test_config['output_filepath'] = 'test1_output.json'
    else:
        test_config['output_filepath'] = name_prefix + '_test1_output.json'

    test_module_config = ModelEvaluatorConfig.parse_obj(test_config) 
    eval_results = ModelEvaluator.from_config(test_module_config)
    
    testMAPE = 0
    testRMSLE = 0
    for metric in eval_results:
        if metric['metric_name'] == 'mape':
            testMAPE += metric['value']
        if metric['metric_name'] == 'rmsle':
            testRMSLE += metric['value']

    metrics['TestMAPE'] = testMAPE
    metrics['TestRMLSE'] = testRMSLE
    metrics.update(parse_metrics(eval_results, 'Test'))
    
    final_train_config = deepcopy(default_train_config)
    final_train_config['data_source'] = data_source
    final_train_config['region_name'] = region
    final_train_config['region_type'] = region_type
    final_train_config['train_start_date'] = train2_start_date
    final_train_config['train_end_date'] = train2_end_date
    final_train_config['search_parameters']['max_evals'] = max_evals
    final_train_config['input_filepath'] = input_filepath
    
    if mlflow_log:
        final_train_config['output_filepath'] = 'train2_output.json'
    else:
        final_train_config['output_filepath'] = name_prefix + '_train2_output.json'

    final_train_module_config = TrainingModuleConfig.parse_obj(final_train_config)
    final_results = TrainingModule.from_config(final_train_module_config)

    train2MAPE = 0
    train2RMSLE = 0
    for metric in final_results['train_metric_results']:
        if metric['metric_name'] == 'mape':
            train2MAPE += metric['value']
        if metric['metric_name'] == 'rmsle':
            train2RMSLE += metric['value']

    metrics['Train2MAPE'] = train2MAPE
    metrics['Train2RMLSE'] = train2RMSLE
    metrics.update(parse_metrics(final_results['train_metric_results'], 'Train2'))
    
    model_params['model_parameters'] = final_results['model_parameters']
    train2_model_params['model_parameters'] = final_results['model_parameters']
    
    return params, metrics, train1_model_params, train2_model_params


def forecast(model_params, run_day, forecast_start_date, forecast_end_date, default_forecast_config,
             with_uncertainty=False, include_best_fit=False):
    """Generate forecasts for a chosen interval using model parameters

    Args:
        model_params (dict): Model parameters
        run_day (str): date to initialize model parameters
        forecast_start_date (str): start date of forecast
        forecast_end_date (str): end date of forecast
        default_forecast_config (dict): default forecast configuration
        with_uncertainty (bool, optional): if True, forecast with uncertainty
        include_best_fit (bool, optional): if True, include best fit forecast

    Returns:
        pd.DataFrame : dataframe containing forecasts
    """

    eval_config = ForecastingModuleConfig.parse_obj(default_forecast_config)
    eval_config.data_source = model_params['data_source']
    eval_config.region_name = model_params['region']
    eval_config.region_type = model_params['region_type']
    eval_config.model_parameters = model_params['model_parameters']
    eval_config.input_filepath = model_params['input_filepath']

    if with_uncertainty:
        eval_config.model_parameters['modes']['predict_mode'] = 'predictions_with_uncertainty'
        eval_config.model_parameters['uncertainty_parameters'] = \
            default_forecast_config['model_parameters']['uncertainty_parameters']

    eval_config.run_day = run_day
    eval_config.forecast_start_date = forecast_start_date
    eval_config.forecast_end_date = forecast_end_date

    forecast_df = ForecastingModule.from_config(eval_config)

    forecast_df_best_fit = pd.DataFrame()
    if include_best_fit:
        eval_config.model_parameters['modes']['predict_mode'] = 'best_fit'
        forecast_df_best_fit = ForecastingModule.from_config(eval_config)
        forecast_df_best_fit = forecast_df_best_fit.drop(columns=['Region Type', 'Region', 'Country', 'Lat', 'Long'])
        for col in forecast_df_best_fit.columns:
            if col.endswith('_mean'):
                new_col = '_'.join([col.split('_')[0], 'best'])
                forecast_df_best_fit = forecast_df_best_fit.rename(columns={col: new_col})
            else:
                forecast_df_best_fit = forecast_df_best_fit.rename(columns={col: '_'.join([col, 'best'])})

    forecast_df = forecast_df.drop(columns=['Region Type', 'Region', 'Country', 'Lat', 'Long'])
    forecast_df = pd.concat([forecast_df_best_fit, forecast_df], axis=1)
    forecast_df = forecast_df.reset_index()
    return forecast_df


def get_observations_in_range(data_source, region_name, region_type, 
                              start_date, end_date,
                              obs_type='confirmed'):
    """
        Return a list of counts of obs_type cases
        from the region in the specified date range.
    """
    
    observations = DataFetcherModule.get_observations_for_region(region_type, region_name, data_source=data_source)
    observations_df = observations[observations['observation'] == obs_type]
    
    start_date = datetime.strptime(start_date, '%m/%d/%y')
    end_date = datetime.strptime(end_date, '%m/%d/%y')
    delta = (end_date - start_date).days
    days = []
    for i in range(delta + 1):
        days.append((start_date + timedelta(days=i)).strftime('%-m/%-d/%-y'))
    
    # Fetch observations in the date range
    observations_df = observations_df[days]
    
    # Transpose the df to get the
    # observations_df.shape = (num_days, 1)
    observations_df = observations_df.reset_index(drop=True).transpose()
    
    # Rename the column to capture observation type
    # Note that the hardcoded 0 in the rename comes from the reset_index
    # from the previous step
    observations = observations_df[0].to_list()
    return observations


def train_eval_forecast(region, region_type,
                        train1_start_date, train1_end_date,
                        train2_start_date, train2_end_date,
                        test_run_day, test_start_date, test_end_date,
                        forecast_run_day, forecast_start_date, forecast_end_date,
                        default_train_config, default_test_config,
                        default_forecast_config, max_evals=1000,
                        data_source=None, input_filepath=None, mlflow_log=True, name_prefix=None,
                        plot_actual_vs_predicted=False, plot_name='default.png'):
    """
        Run train, evaluation and generate forecasts as a dataframe.

        If plot_actual_vs_predicted is set to True,
        we first check if the forecast_end_date is prior to the current date
        so that we have actual_confirmed cases and then plot the predictions.
    """
    
    params, metrics, model_params = train_eval(region, region_type,
                                               train1_start_date, train1_end_date,
                                               train2_start_date, train2_end_date,
                                               test_run_day, test_start_date, test_end_date,
                                               default_train_config, default_test_config,
                                               max_evals=max_evals, data_source=data_source,
                                               input_filepath=input_filepath,
                                               mlflow_log=mlflow_log, name_prefix=name_prefix)
    model_params['model_parameters']['incubation_period'] = 5
    forecast_df = forecast(model_params, forecast_run_day, 
                           forecast_start_date, forecast_end_date, 
                           default_forecast_config)

    plot(model_params, forecast_df, forecast_start_date,
         forecast_end_date, plot_name=plot_name)

    return forecast_df, params, metrics, model_params


def set_dates(current_day, train_period=7, test_period=7):
    
    dates = dict()

    # Train1
    train1_start_date = current_day - timedelta(train_period + test_period - 1)
    train1_end_date = current_day - timedelta(test_period)
    train1_run_day = train1_start_date - timedelta(1)

    dates['train1_start_date'] = train1_start_date.strftime("%-m/%-d/%y")
    dates['train1_end_date'] = train1_end_date.strftime("%-m/%-d/%y")
    dates['train1_run_day'] = train1_run_day.strftime("%-m/%-d/%y")

    # Train2
    train2_start_date = current_day - timedelta(train_period - 1)
    train2_end_date = current_day
    train2_run_day = train2_start_date - timedelta(1)

    dates['train2_start_date'] = train2_start_date.strftime("%-m/%-d/%y")
    dates['train2_end_date'] = train2_end_date.strftime("%-m/%-d/%y")
    dates['train2_run_day'] = train2_run_day.strftime("%-m/%-d/%y")

    # Test1
    test_start_date = current_day - timedelta(test_period - 1)
    test_end_date = current_day
    test_run_day = test_start_date - timedelta(1)

    dates['test_start_date'] = test_start_date.strftime("%-m/%-d/%y")
    dates['test_end_date'] = test_end_date.strftime("%-m/%-d/%y")
    dates['test_run_day'] = test_run_day.strftime("%-m/%-d/%y")

    return dates


def train_eval_plot(region, region_type, 
                    current_day, forecast_length,
                    default_train_config, default_test_config, default_forecast_config,
                    max_evals=1000, data_source=None, input_filepath=None,
                    mlflow_log=False, mlflow_run_name=None):
    """
        Run train, evaluation and plotting. 
    
    Arguments:
        region, region_type : Region info corresponding to the run
        current_day : Indicates end of test/train2 interval
        forecast_length : Length of forecast interval
        default_train_config : Default train config (loaded from train_config.json)
        default_test_config : Default test config (loaded from test_config.json)
        max_evals : number of search evaluations for SEIR (default: 1000)
        data_source : Data source for picking the region data
        mlflow_log : Experiment logged using MLFlow (default: True)
        mlflow_run_name : Name given to run in MLFlow (default: None)

    Note:
        date_format : datetime.date object 
    """
        
    name_prefix = " ".join(region)
        
    dates = set_dates(current_day)
    
    train1_start_date = dates['train1_start_date']
    train1_end_date = dates['train1_end_date']
    train1_run_day = dates['train1_run_day']
    
    train2_start_date = dates['train2_start_date']
    train2_end_date = dates['train2_end_date']
    train2_run_day = dates['train2_run_day']
    
    test_start_date = dates['test_start_date']
    test_end_date = dates['test_end_date']
    test_run_day = dates['test_run_day']
    
    params, metrics, train1_params, train2_params = train_eval(region, region_type,
                                                               train1_start_date, train1_end_date,
                                                               train2_start_date, train2_end_date, train2_run_day,
                                                               test_start_date, test_end_date,
                                                               default_train_config, default_test_config,
                                                               max_evals=max_evals, data_source=data_source,
                                                               input_filepath=input_filepath,
                                                               mlflow_log=mlflow_log,
                                                               name_prefix=name_prefix)

    plot_m1(train1_params, train1_run_day, train1_start_date, train1_end_date, test_run_day, test_start_date,
            test_end_date, default_forecast_config, plot_name=name_prefix + '_m1.png')

    plot_m2(train2_params, train1_start_date, train1_end_date, test_run_day, test_start_date, test_end_date,
            default_forecast_config, plot_name=name_prefix + '_m2.png')

    forecast_start_date = (datetime.strptime(train2_end_date, "%m/%d/%y") + timedelta(1)).strftime("%-m/%-d/%y")
    plot_m3(train2_params, train1_start_date, forecast_start_date, forecast_length, default_forecast_config,
            plot_name=name_prefix + '_m3.png')
    
    if mlflow_log:
        with mlflow.start_run(run_name=mlflow_run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(name_prefix+'_m1.png')
            mlflow.log_artifact(name_prefix+'_m2.png')
            mlflow.log_artifact(name_prefix+'_m3.png')
            mlflow.log_artifact('train_config.json')
            mlflow.log_artifact('train1_output.json')
            mlflow.log_artifact('test1_output.json')
            mlflow.log_artifact('train2_output.json')


def train_eval_ensemble(region, region_type,
                        train1_start_date, train1_end_date,
                        train2_start_date, train2_end_date, run_day,
                        test_start_date, test_end_date,
                        default_train_config, default_test_config,
                        max_evals=1000, data_source=None, input_filepath=None,
                        mlflow_log=True, name_prefix=None):

    params = dict()
    metrics = dict()

    # params['region'] = " ".join(region)
    # params['region_type'] = region_type
    # params['train1_start_date'] = train1_start_date
    # params['train1_end_date'] = train1_end_date
    # params['train2_start_date'] = train2_start_date
    # params['train2_end_date'] = train2_end_date
    # params['run_day'] = run_day
    # params['test_start_date'] = test_start_date
    # params['test_end_date'] = test_end_date
    # params['data_source'] = data_source

    train_config = deepcopy(default_train_config)
    train_config['data_source'] = data_source
    train_config['region_name'] = region
    train_config['region_type'] = region_type
    train_config['train_start_date'] = train1_start_date
    train_config['train_end_date'] = train1_end_date
    train_config['search_parameters']['max_evals'] = max_evals
    train_config['model_parameters']['modes']['training_mode'] = 'full'
    train_config['input_filepath'] = input_filepath

    # model parameters
    model_params = dict()
    model_params['data_source'] = data_source
    model_params['region'] = region
    model_params['region_type'] = region_type
    model_params['model_type'] = train_config['model_class']
    model_params['input_filepath'] = input_filepath

    train1_model_params = deepcopy(model_params)
    train2_model_params = deepcopy(model_params)

    if mlflow_log:
        train_config['output_filepath'] = 'train1_output.json'
    else:
        assert name_prefix is not None
        train_config['output_filepath'] = name_prefix + '_train1_output.json'

    print("Performing M1 fit...")
    train_module_config = TrainingModuleConfig.parse_obj(train_config)
    train_results = TrainingModule.from_config(train_module_config)

    train1_model_params['model_parameters'] = train_results['model_parameters']
    metrics['M1_beta'] = train_results['model_parameters']['beta']
    metrics_m1_train_losses = loss_json_to_dataframe(train_results['train_metric_results'], 'train1')

    test_config = deepcopy(default_test_config)
    test_config['data_source'] = data_source
    test_config['region_name'] = region
    test_config['region_type'] = region_type
    test_config['test_start_date'] = test_start_date
    test_config['test_end_date'] = test_end_date
    test_config['run_day'] = run_day
    test_config['model_parameters'].update(train_results['model_parameters'])
    test_config['model_parameters']['modes']['predict_mode'] = 'mean_predictions'
    test_config['input_filepath'] = input_filepath

    if mlflow_log:
        test_config['output_filepath'] = 'test1_output.json'
    else:
        test_config['output_filepath'] = name_prefix + '_test1_output.json'

    print("Evaluating M1 model...")
    test_module_config = ModelEvaluatorConfig.parse_obj(test_config)
    eval_results = ModelEvaluator.from_config(test_module_config)
    metrics_m1_test_losses = loss_json_to_dataframe(eval_results, 'test1')
    metrics['M1_losses'] = pd.concat([metrics_m1_train_losses, metrics_m1_test_losses], axis=0)

    testMAPE = 0
    testRMSLE = 0
    for metric in eval_results:
        if metric['metric_name'] == 'mape':
            testMAPE += metric['value']
        if metric['metric_name'] == 'rmsle':
            testRMSLE += metric['value']

    metrics['test1_MAPE'] = testMAPE
    metrics['test1_RMLSE'] = testRMSLE
    metrics.update(parse_metrics(eval_results, 'test1'))

    final_train_config = deepcopy(default_train_config)
    final_train_config['data_source'] = data_source
    final_train_config['region_name'] = region
    final_train_config['region_type'] = region_type
    final_train_config['train_start_date'] = train2_start_date
    final_train_config['train_end_date'] = train2_end_date
    final_train_config['search_parameters']['max_evals'] = max_evals
    final_train_config['model_parameters']['modes']['training_mode'] = 'full'
    final_train_config['input_filepath'] = input_filepath

    if mlflow_log:
        final_train_config['output_filepath'] = 'train2_output.json'
    else:
        final_train_config['output_filepath'] = name_prefix + '_train2_output.json'

    print("Performing M2 fit...")
    final_train_module_config = TrainingModuleConfig.parse_obj(final_train_config)
    final_results = TrainingModule.from_config(final_train_module_config)

    model_params['model_parameters'] = final_results['model_parameters']
    train2_model_params['model_parameters'] = final_results['model_parameters']
    metrics['M2_beta'] = final_results['model_parameters']['beta']
    metrics['M2_losses'] = loss_json_to_dataframe(final_results['train_metric_results'], 'train2')

    return params, metrics, train1_model_params, train2_model_params


def train_eval_plot_ensemble(region, region_type,
                             train1_run_day, train1_start_date, train1_end_date,
                             test_run_day, test_start_date, test_end_date,
                             train2_run_day, train2_start_date, train2_end_date,
                             forecast_run_day, forecast_start_date, forecast_end_date,
                             default_train_config, default_test_config, default_forecast_config,
                             child_model_max_eval=10, ensemble_model_max_eval=10, data_source=None,
                             input_filepath=None, output_dir='', mlflow_log=False, mlflow_run_name=None):

    params_dict = dict()
    metrics_dict = dict()
    artifacts_dict = dict()
    dates = dict()

    name_prefix = " ".join(region[0])

    #Persist the training/testing durations
    dates['train1_run_day'] = train1_run_day
    dates['train1_start_date'] = train1_start_date
    dates['train1_end_date'] = train1_end_date

    dates['train2_run_day'] = train2_run_day
    dates['train2_start_date'] = train2_start_date
    dates['train2_end_date'] = train2_end_date

    dates['test_run_day'] = test_run_day
    dates['test_start_date'] = test_start_date
    dates['test_end_date'] = test_end_date

    dates['forecast_run_day'] = forecast_run_day
    dates['forecast_start_date'] = forecast_start_date
    dates['forecast_end_date'] = forecast_end_date

    params_dict.update({'time_interval_config': dates})
    params_dict['region_name'] = " ".join(region)
    params_dict['region_type'] = region_type
    params_dict['data_source'] = data_source
    params_dict['data_file_path'] = os.path.join('file:///', os.getcwd(), input_filepath) \
        if input_filepath is not None else "No path available"
    params_dict['search_parameters'] = default_train_config['search_parameters']
    params_dict['param_search_space_config'] = default_train_config['search_space']
    params_dict['train_loss_function_config'] = default_train_config['training_loss_function']
    params_dict['eval_loss_function_config'] = default_train_config['loss_functions']
    uncertainty_params = default_forecast_config['model_parameters']['uncertainty_parameters']
    params_dict['forecast_percentiles'] = uncertainty_params['percentiles']
    params_dict['forecast_planning_variable'] = uncertainty_params['column_of_interest']
    params_dict['forecast_planning_date'] = uncertainty_params['date_of_interest']

    # TODO: Fix this param
    max_evals = 10

    params, metrics, train1_params, train2_params = train_eval_ensemble([region], region_type,
                                                                        train1_start_date, train1_end_date,
                                                                        train2_start_date, train2_end_date,
                                                                        train2_run_day,
                                                                        test_start_date, test_end_date,
                                                                        default_train_config, default_test_config,
                                                                        max_evals=max_evals, data_source=data_source,
                                                                        input_filepath=input_filepath,
                                                                        mlflow_log=mlflow_log,
                                                                        name_prefix=name_prefix)

    print('forecast_config')
    print(default_forecast_config)
    print(default_forecast_config['forecast_start_date'])
    print(type(default_forecast_config['forecast_start_date']))
    print(default_forecast_config['forecast_end_date'])

    print("Creating artifacts...")
    create_plots(region, region_type, train1_params, train2_params, train1_run_day, train1_start_date, train1_end_date,
                 test_run_day, test_start_date, test_end_date, train2_run_day, train2_start_date, train2_end_date,
                 forecast_run_day, forecast_start_date, forecast_end_date, default_forecast_config,
                 data_source=data_source, input_filepath=input_filepath, output_dir=output_dir, debug=True)
   
    if mlflow_log:
        print("Logging to MLflow...")
        with mlflow.start_run(run_name=mlflow_run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(name_prefix+'_m1.png')
            mlflow.log_artifact(name_prefix+'_m2.png')
            mlflow.log_artifact(name_prefix+'_m3.png')
            mlflow.log_artifact('train_config.json')
            mlflow.log_artifact('train1_output.json')
            mlflow.log_artifact('test1_output.json')
            mlflow.log_artifact('train2_output.json')

    artifacts_dict = {
        'plot_M1_CARD': os.path.join(output_dir, 'm1.png'),
        'plot_M1_single_C': os.path.join(output_dir, 'm1_confirmed.png'),
        'plot_M1_single_A': os.path.join(output_dir, 'm1_hospitalized.png'),
        'plot_M1_single_R': os.path.join(output_dir, 'm1_recovered.png'),
        'plot_M1_single_D': os.path.join(output_dir, 'm1_deceased.png'),
        'plot_M2_CARD': os.path.join(output_dir, 'm2.png'),
        'plot_M2_single_C': os.path.join(output_dir, 'm2_confirmed.png'),
        'plot_M2_single_A': os.path.join(output_dir, 'm2_hospitalized.png'),
        'plot_M2_single_R': os.path.join(output_dir, 'm2_recovered.png'),
        'plot_M2_single_D': os.path.join(output_dir, 'm2_deceased.png'),
        'plot_M2_forecast_CARD': os.path.join(output_dir, 'm2_forecast.png'),
        'plot_M2_forecast_single_C': os.path.join(output_dir, 'm2_forecast_confirmed.png'),
        'plot_M2_forecast_single_A': os.path.join(output_dir, 'm2_forecast_hospitalized.png'),
        'plot_M2_forecast_single_R': os.path.join(output_dir, 'm2_forecast_recovered.png'),
        'plot_M2_forecast_single_D': os.path.join(output_dir, 'm2_forecast_deceased.png'),
        'plot_planning_pdf_cdf': os.path.join(output_dir, 'm2_distribution.png'),
        'output_forecast_file': os.path.join(output_dir, 'forecast.csv')
    }

    return params_dict, metrics, artifacts_dict, train1_params, train2_params


def train_eval_plot_ensemble_v1(region, region_type,
                                current_day, forecast_length,
                                default_train_config, default_test_config, default_forecast_config,
                                train_period=14, test_period=7, max_evals=1000, data_source=None,
                                input_filepath=None, output_dir='', mlflow_log=False, mlflow_run_name=None):

    params_dict = dict()
    metrics_dict = dict()
    artifacts_dict = dict()

    name_prefix = " ".join(region)

    dates = set_dates(current_day, train_period, test_period)

    train1_start_date = dates['train1_start_date']
    train1_end_date = dates['train1_end_date']
    train1_run_day = dates['train1_run_day']

    train2_start_date = dates['train2_start_date']
    train2_end_date = dates['train2_end_date']
    train2_run_day = dates['train2_run_day']

    test_start_date = dates['test_start_date']
    test_end_date = dates['test_end_date']
    test_run_day = dates['test_run_day']

    # Set forecast dates
    forecast_start_date = (datetime.strptime(train2_end_date, "%m/%d/%y") + timedelta(1)).strftime("%-m/%-d/%y")
    forecast_run_day = (datetime.strptime(forecast_start_date, "%m/%d/%y") - timedelta(days=1)).strftime("%-m/%-d/%y")
    forecast_end_date = (
            datetime.strptime(forecast_start_date, "%m/%d/%y") + timedelta(days=forecast_length)).strftime("%-m/%-d/%y")

    dates['forecast_start_date'] = forecast_start_date
    dates['forecast_run_day'] = forecast_run_day
    dates['forecast_end_date'] = forecast_end_date

    params_dict.update({'time_interval_config': dates})
    params_dict['region_name'] = " ".join(region)
    params_dict['region_type'] = region_type
    params_dict['data_source'] = data_source
    params_dict['data_file_path'] = os.path.join('file:///', os.getcwd(), input_filepath) \
        if input_filepath is not None else "No path available"
    params_dict['search_parameters'] = default_train_config['search_parameters']
    params_dict['param_search_space_config'] = default_train_config['search_space']
    params_dict['train_loss_function_config'] = default_train_config['training_loss_function']
    params_dict['eval_loss_function_config'] = default_train_config['loss_functions']
    uncertainty_params = default_forecast_config['model_parameters']['uncertainty_parameters']
    params_dict['forecast_percentiles'] = uncertainty_params['percentiles']
    params_dict['forecast_planning_variable'] = uncertainty_params['column_of_interest']
    params_dict['forecast_planning_date'] = uncertainty_params['date_of_interest']

    params, metrics, train1_params, train2_params = train_eval_ensemble(region, region_type,
                                                                        train1_start_date, train1_end_date,
                                                                        train2_start_date, train2_end_date,
                                                                        train2_run_day,
                                                                        test_start_date, test_end_date,
                                                                        default_train_config, default_test_config,
                                                                        max_evals=max_evals, data_source=data_source,
                                                                        input_filepath=input_filepath,
                                                                        mlflow_log=mlflow_log,
                                                                        name_prefix=name_prefix)

    print('forecast_config')
    print(default_forecast_config)
    print(default_forecast_config['forecast_start_date'])
    print(type(default_forecast_config['forecast_start_date']))
    print(default_forecast_config['forecast_end_date'])

    print("Creating artifacts...")
    create_plots(region, region_type, train1_params, train2_params, train1_run_day, train1_start_date, train1_end_date,
                 test_run_day, test_start_date, test_end_date, train2_run_day, train2_start_date, train2_end_date,
                 forecast_run_day, forecast_start_date, forecast_end_date, default_forecast_config,
                 data_source=data_source, input_filepath=input_filepath, output_dir=output_dir, debug=True)
   
    if mlflow_log:
        print("Logging to MLflow...")
        with mlflow.start_run(run_name=mlflow_run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(name_prefix+'_m1.png')
            mlflow.log_artifact(name_prefix+'_m2.png')
            mlflow.log_artifact(name_prefix+'_m3.png')
            mlflow.log_artifact('train_config.json')
            mlflow.log_artifact('train1_output.json')
            mlflow.log_artifact('test1_output.json')
            mlflow.log_artifact('train2_output.json')

    artifacts_dict = {
        'plot_M1_CARD': os.path.join(output_dir, 'm1.png'),
        'plot_M1_single_C': os.path.join(output_dir, 'm1_confirmed.png'),
        'plot_M1_single_A': os.path.join(output_dir, 'm1_hospitalized.png'),
        'plot_M1_single_R': os.path.join(output_dir, 'm1_recovered.png'),
        'plot_M1_single_D': os.path.join(output_dir, 'm1_deceased.png'),
        'plot_M2_CARD': os.path.join(output_dir, 'm2.png'),
        'plot_M2_single_C': os.path.join(output_dir, 'm2_confirmed.png'),
        'plot_M2_single_A': os.path.join(output_dir, 'm2_hospitalized.png'),
        'plot_M2_single_R': os.path.join(output_dir, 'm2_recovered.png'),
        'plot_M2_single_D': os.path.join(output_dir, 'm2_deceased.png'),
        'plot_M2_forecast_CARD': os.path.join(output_dir, 'm2_forecast.png'),
        'plot_M2_forecast_single_C': os.path.join(output_dir, 'm2_forecast_confirmed.png'),
        'plot_M2_forecast_single_A': os.path.join(output_dir, 'm2_forecast_hospitalized.png'),
        'plot_M2_forecast_single_R': os.path.join(output_dir, 'm2_forecast_recovered.png'),
        'plot_M2_forecast_single_D': os.path.join(output_dir, 'm2_forecast_deceased.png'),
        'plot_planning_pdf_cdf': os.path.join(output_dir, 'm2_distribution.png'),
        'output_forecast_file': os.path.join(output_dir, 'forecast.csv')
    }

    return params_dict, metrics, artifacts_dict, train1_params, train2_params


def add_init_observations_to_predictions(df_actual, df_predictions, run_day):

    init_observations = get_observations_subset(df_actual, run_day, run_day)
    init_observations_df = pd.DataFrame(columns=df_predictions.columns)
    for col in init_observations_df.columns:
        original_col = col.split('_')[0]
        if original_col in init_observations:
            init_observations_df.loc[:, col] = init_observations.loc[:, original_col]
    init_observations_df.loc[:, 'date'] = init_observations.loc[:, 'index'].apply(lambda d: d.strftime('%-m/%-d/%-y'))
    init_observations_df.fillna(0, inplace=True)
    df_predictions = pd.concat([init_observations_df, df_predictions], axis=0, ignore_index=True)
    return df_predictions


def get_observations_subset(df_actual, start_date=None, end_date=None):

    df_actual = df_actual.set_index('observation')
    df_actual = df_actual.transpose().reset_index()
    if start_date is not None:
        start = df_actual.index[df_actual['index'] == start_date].tolist()[0]
    else:
        start = df_actual.index.min()
    if end_date is not None:
        end = df_actual.index[df_actual['index'] == end_date].tolist()[0]
    else:
        end = df_actual.index.max()
    df_actual = df_actual[start: end + 1]
    df_actual['index'] = pd.to_datetime(df_actual['index'])

    return df_actual
    

def create_plots(region, region_type, train1_model_params, train2_model_params,
                 train1_run_day, train1_start_date, train1_end_date,
                 test_run_day, test_start_date, test_end_date,
                 train2_run_day, train2_start_date, train2_end_date,
                 forecast_run_day, forecast_start_date, forecast_end_date, forecast_config,
                 data_source=None, input_filepath=None, output_dir='', debug=False):
    # TODO: Accept plot titles/paths as params

    # Get actual and smoothed observations
    df_actual = DataFetcherModule.get_observations_for_region(region_type, region, data_source=data_source,
                                                              smooth=False, filepath=input_filepath)
    df_smoothed = DataFetcherModule.get_observations_for_region(region_type, region, data_source=data_source,
                                                                smooth=True, filepath=input_filepath)

    # Set start date of plots
    plot_start_date_m1 = (datetime.strptime(train1_start_date, "%m/%d/%y") - timedelta(days=7)).strftime("%-m/%-d/%y")
    plot_start_date_m2 = (datetime.strptime(train2_start_date, "%m/%d/%y") - timedelta(days=7)).strftime("%-m/%-d/%y")

    # Get actual and smoothed observations in the correct ranges
    df_actual_m1 = get_observations_subset(df_actual, plot_start_date_m1, test_end_date)
    df_smoothed_m1 = get_observations_subset(df_smoothed, plot_start_date_m1, test_end_date)
    df_actual_m2 = get_observations_subset(df_actual, plot_start_date_m2, train2_end_date)
    df_smoothed_m2 = get_observations_subset(df_smoothed, plot_start_date_m2, train2_end_date)

    # Get predictions for M1 train and test intervals
    # Get train predictions for M1, add run day observations and convert the date column to datetime
    # Get test predictions for M1 until the end of the forecast interval to include planning date for uncertainty
    # Retain only predictions in test range, add run day observations and convert the date column to datetime
    # M1 train
    if debug:
        df_predictions_train_m1 = forecast(train1_model_params, train1_run_day, train1_start_date, forecast_end_date,
                                           forecast_config, with_uncertainty=True, include_best_fit=True)
        start_date, end_date = datetime.strptime(train1_start_date, '%m/%d/%y'), datetime.strptime(train1_end_date,
                                                                                                   '%m/%d/%y')
        delta = (end_date - start_date).days
        days = []
        for i in range(delta + 1):
            days.append((start_date + timedelta(days=i)).strftime('%-m/%-d/%-y'))
        df_predictions_train_m1 = df_predictions_train_m1.set_index('date').loc[days].reset_index()
        df_predictions_train_m1 = add_init_observations_to_predictions(df_actual, df_predictions_train_m1,
                                                                       train1_run_day)
        df_predictions_train_m1['date'] = pd.to_datetime(df_predictions_train_m1['date'])
    else:
        df_predictions_train_m1 = forecast(train1_model_params, train1_run_day, train1_start_date, train1_end_date,
                                           forecast_config)
        df_predictions_train_m1 = add_init_observations_to_predictions(df_actual, df_predictions_train_m1,
                                                                       train1_run_day)
        df_predictions_train_m1['date'] = pd.to_datetime(df_predictions_train_m1['date'])

    # M1 test
    df_predictions_test_m1 = forecast(train1_model_params, test_run_day, test_start_date, forecast_end_date,
                                      forecast_config, with_uncertainty=True, include_best_fit=True)
    start_date, end_date = datetime.strptime(test_start_date, '%m/%d/%y'), datetime.strptime(test_end_date, '%m/%d/%y')
    delta = (end_date - start_date).days
    days = []
    for i in range(delta + 1):
        days.append((start_date + timedelta(days=i)).strftime('%-m/%-d/%-y'))
    df_predictions_test_m1 = df_predictions_test_m1.set_index('date').loc[days].reset_index()
    df_predictions_test_m1 = add_init_observations_to_predictions(df_actual, df_predictions_test_m1, test_run_day)
    df_predictions_test_m1['date'] = pd.to_datetime(df_predictions_test_m1['date'])

    # Get predictions for M2 train and forecast intervals
    # Get train predictions for M2, add run day observations and convert the date column to datetime
    # Get forecast predictions for M2, add run day observations and convert the date column to datetime

    # M2 train
    if debug:
        df_predictions_train_m2 = forecast(train2_model_params, train2_run_day, train2_start_date, forecast_end_date,
                                           forecast_config, with_uncertainty=True, include_best_fit=True)
        start_date, end_date = datetime.strptime(train2_start_date, '%m/%d/%y'), datetime.strptime(train2_end_date,
                                                                                                   '%m/%d/%y')
        delta = (end_date - start_date).days
        days = []
        for i in range(delta + 1):
            days.append((start_date + timedelta(days=i)).strftime('%-m/%-d/%-y'))
        df_predictions_train_m2 = df_predictions_train_m2.set_index('date').loc[days].reset_index()
        df_predictions_train_m2 = add_init_observations_to_predictions(df_actual, df_predictions_train_m2,
                                                                       train2_run_day)
        df_predictions_train_m2['date'] = pd.to_datetime(df_predictions_train_m2['date'])
    else:
        df_predictions_train_m2 = forecast(train2_model_params, train2_run_day, train2_start_date, train2_end_date,
                                           forecast_config)
        df_predictions_train_m2 = add_init_observations_to_predictions(df_actual, df_predictions_train_m2,
                                                                       train2_run_day)
        df_predictions_train_m2['date'] = pd.to_datetime(df_predictions_train_m2['date'])

    # M2 forecast
    df_predictions_forecast_m2 = forecast(train2_model_params, forecast_run_day, forecast_start_date, forecast_end_date,
                                          forecast_config, with_uncertainty=True, include_best_fit=True)
    df_predictions_forecast_m2.to_csv(os.path.join(output_dir, 'forecast.csv'))
    df_predictions_forecast_m2 = add_init_observations_to_predictions(df_actual, df_predictions_forecast_m2,
                                                                      forecast_run_day)
    df_predictions_forecast_m2['date'] = pd.to_datetime(df_predictions_forecast_m2['date'])

    # Get percentiles to be plotted
    uncertainty_params = forecast_config['model_parameters']['uncertainty_parameters']
    confidence_intervals = []
    for c in uncertainty_params['ci']:
        confidence_intervals.extend([50 - c / 2, 50 + c / 2])
    percentiles = list(set(uncertainty_params['percentiles'] + confidence_intervals))
    column_tags = [str(i) for i in percentiles]
    column_tags.extend(['mean', 'best'])
    column_of_interest = uncertainty_params['column_of_interest']

    region_name = " ".join(region)

    # Create M1, M2, M2 forecast plots
    m1_plots(region_name, df_actual_m1, df_smoothed_m1, df_predictions_train_m1, df_predictions_test_m1,
             train1_start_date, test_start_date, column_tags=column_tags, output_dir=output_dir, debug=debug)
    m2_plots(region_name, df_actual_m2, df_smoothed_m2, df_predictions_train_m2, train2_start_date,
             column_tags=column_tags, output_dir=output_dir, debug=debug)
    m2_forecast_plots(region_name, df_actual_m2, df_smoothed_m2, df_predictions_forecast_m2,
                      train2_start_date, forecast_start_date, column_tags=column_tags, output_dir=output_dir,
                      debug=False)

    # Get trials dataframe
    region_metadata = DataFetcherModule.get_regional_metadata(region_type, region, data_source=data_source)
    model = ModelFactory.get_model(train1_model_params['model_type'], train2_model_params['model_parameters'])
    trials = model.get_trials_distribution(region_metadata, df_actual, forecast_run_day, forecast_start_date,
                                           forecast_end_date)
    # Plot PDF and CDF
    distribution_plots(trials, column_of_interest, output_dir=output_dir)


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

    # TODO: Log scale
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
        train1_model_params['region_type'], train1_model_params['region'],
        data_source=train1_model_params['data_source'], filepath=train1_model_params['input_filepath'])
    actual = actual.set_index('observation')
    actual = actual.transpose()
    actual = actual.reset_index()
    start = actual.index[actual['index'] == actual_start_date].tolist()[0]
    end = actual.index[actual['index'] == test_end_date].tolist()[0]
    actual = actual[start: end + 1]
    actual['index'] = pd.to_datetime(actual['index'])

    plot_markers = plot_config['markers']
    plot_colors = plot_config['colors']
    plot_labels = plot_config['labels']
    plot_variables = plot_config['variables']

    fig, ax = plt.subplots(figsize=(16, 12))

    for variable in plot_variables:

        # Plot observed values
        ax.plot(actual['index'], actual[variable], plot_markers['observed'],
                color=plot_colors[variable], label=plot_labels[variable] + ': Observed')

        for pd_df in [pd_df_train, pd_df_test]:

            # Plot mean predictions
            if variable + '_mean' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable + '_mean'], plot_markers['predicted']['mean'],
                        color=plot_colors[variable], label=plot_labels[variable] + ': Predicted')

            # Plot uncertainty in predictions
            if plot_config['uncertainty']:

                if variable + '_min' in pd_df:
                    ax.plot(pd_df['index'], pd_df[variable + '_min'], plot_markers['predicted']['min'],
                            color=plot_colors[variable], label=plot_labels[variable] + ': Predicted (Min)')

                if variable + '_max' in pd_df:
                    ax.plot(pd_df['index'], pd_df[variable + '_max'], plot_markers['predicted']['max'],
                            color=plot_colors[variable], label=plot_labels[variable] + ': Predicted (Max)')

            # Plot rolling average
            if plot_config['rolling_average'] and variable + '_ra' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable + '_ra'], plot_markers['rolling_average'],
                        color=plot_colors[variable], label=plot_labels[variable] + ': Predicted (RA)')

    train_start = pd.to_datetime(train1_start_date)
    test_start = pd.to_datetime(test_start_date)

    line_height = plt.ylim()[1]
    ax.plot([train_start, train_start], [0, line_height], '--', color='brown', label='Train starts')
    ax.plot([test_start, test_start], [0, line_height], '--', color='black', label='Test starts')

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

    # TODO: Log scale
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
        train2_model_params['region_type'], train2_model_params['region'],
        data_source=train2_model_params['data_source'], filepath=train2_model_params['input_filepath'])
    actual = actual.set_index('observation')
    actual = actual.transpose()
    actual = actual.reset_index()
    start = actual.index[actual['index'] == actual_start_date].tolist()[0]
    end = actual.index[actual['index'] == train2_end_date].tolist()[0]
    actual = actual[start: end + 1]
    actual['index'] = pd.to_datetime(actual['index'])

    plot_markers = plot_config['markers']
    plot_colors = plot_config['colors']
    plot_labels = plot_config['labels']
    plot_variables = plot_config['variables']

    fig, ax = plt.subplots(figsize=(16, 12))

    for variable in plot_variables:

        # Plot observed values
        ax.plot(actual['index'], actual[variable], plot_markers['observed'],
                color=plot_colors[variable], label=plot_labels[variable] + ': Observed')

        # Plot mean predictions
        if variable + '_mean' in pd_df:
            ax.plot(pd_df['index'], pd_df[variable + '_mean'], plot_markers['predicted']['mean'],
                    color=plot_colors[variable], label=plot_labels[variable] + ': Predicted')

        # Plot uncertainty in predictions
        if plot_config['uncertainty']:

            if variable + '_min' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable + '_min'], plot_markers['predicted']['min'],
                        color=plot_colors[variable], label=plot_labels[variable] + ': Predicted (Min)')

            if variable + '_max' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable + '_max'], plot_markers['predicted']['max'],
                        color=plot_colors[variable], label=plot_labels[variable] + ': Predicted (Max)')

        # Plot rolling average
        if plot_config['rolling_average'] and variable + '_ra' in pd_df:
            ax.plot(pd_df['index'], pd_df[variable + '_ra'], plot_markers['rolling_average'],
                    color=plot_colors[variable], label=plot_labels[variable] + ': Predicted (RA)')

    train2_start = pd.to_datetime(train2_start_date)

    line_height = plt.ylim()[1]
    ax.plot([train2_start, train2_start], [0, line_height], '--', color='black', label='Train starts')

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

    # TODO: Log scale

    with open(plot_config) as fplot:
        default_plot_config = json.load(fplot)

    plot_config = deepcopy(default_plot_config)
    plot_config['uncertainty'] = uncertainty
    plot_config['rolling_average'] = rolling_average

    actual_start_date = (datetime.strptime(train1_start_date, "%m/%d/%y") - timedelta(days=14)).strftime("%-m/%-d/%y")
    forecast_run_day = (datetime.strptime(forecast_start_date, "%m/%d/%y") - timedelta(days=1)).strftime("%-m/%-d/%y")
    forecast_end_date = (datetime.strptime(forecast_start_date, "%m/%d/%y") + timedelta(days=forecast_length)).strftime(
        "%-m/%-d/%y")

    # Get predictions
    pd_df_forecast = forecast(train2_model_params, forecast_run_day, forecast_start_date, forecast_end_date,
                              forecast_config)

    pd_df_forecast['index'] = pd.to_datetime(pd_df_forecast['index'])
    pd_df = pd_df_forecast.sort_values(by=['index'])

    # Get observed data
    actual = DataFetcherModule.get_observations_for_region(
        train2_model_params['region_type'], train2_model_params['region'],
        data_source=train2_model_params['data_source'], filepath=train2_model_params['input_filepath'])
    actual = actual.set_index('observation')
    actual = actual.transpose()
    actual = actual.reset_index()
    start = actual.index[actual['index'] == actual_start_date].tolist()[0]
    end = actual.index[actual['index'] == forecast_run_day].tolist()[0]
    actual = actual[start: end + 1]
    actual['index'] = pd.to_datetime(actual['index'])

    plot_markers = plot_config['markers']
    plot_colors = plot_config['colors']
    plot_labels = plot_config['labels']
    plot_variables = plot_config['variables']

    fig, ax = plt.subplots(figsize=(16, 12))

    for variable in plot_variables:

        # Plot observed values
        ax.plot(actual['index'], actual[variable], plot_markers['observed'],
                color=plot_colors[variable], label=plot_labels[variable] + ': Observed')

        # Plot mean predictions
        if variable + '_mean' in pd_df:
            ax.plot(pd_df['index'], pd_df[variable + '_mean'], plot_markers['predicted']['mean'],
                    color=plot_colors[variable], label=plot_labels[variable] + ': Predicted')

        # Plot uncertainty in predictions
        if plot_config['uncertainty']:

            if variable + '_min' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable + '_min'], plot_markers['predicted']['min'],
                        color=plot_colors[variable], label=plot_labels[variable] + ': Predicted (Min)')

            if variable + '_max' in pd_df:
                ax.plot(pd_df['index'], pd_df[variable + '_max'], plot_markers['predicted']['max'],
                        color=plot_colors[variable], label=plot_labels[variable] + ': Predicted (Max)')

        # Plot rolling average
        if plot_config['rolling_average'] and variable + '_ra' in pd_df:
            ax.plot(pd_df['index'], pd_df[variable + '_ra'], plot_markers['rolling_average'],
                    color=plot_colors[variable], label=plot_labels[variable] + ': Predicted (RA)')

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


def plot_data(region, region_type, dir_name, dir_prefix='../notebooks',
    data_source=None, data_path=None,
    plot_config='plot_config.json', plot_name='default.png',
    csv_name='csv_cnt_data.csv'):
    with open(plot_config) as fin:
        default_plot_config = json.load(fin)

    plot_config = deepcopy(default_plot_config)

    actual = DataFetcherModule.get_observations_for_region(region_type, region, data_source=data_source,
                                                           filepath=data_path, smooth=False)
    actual.drop(columns=['region_name', 'region_type'], inplace=True)
    actual = actual.set_index('observation').transpose().reset_index()
    actual['index'] = pd.to_datetime(actual['index'])
    actual = actual.loc[~ (actual.select_dtypes(include=['number']) == 0).all(axis='columns'), :]  # CHECK THIS
    csv_path = os.path.join(dir_prefix, dir_name, csv_name)
    actual.to_csv(csv_path, index=False)

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

    plot_path = os.path.join(dir_prefix, dir_name, plot_name)
    plt.savefig(plot_path)
