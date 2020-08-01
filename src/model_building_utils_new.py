import json
import os
from copy import deepcopy
from datetime import datetime

import mlflow
import pandas as pd
import publishers.mlflow_logging as mlflow_logger
import publishers.report_generation as reporting
import utils.staffing as domain_info
from configs.base_config import TrainingModuleConfig, ModelEvaluatorConfig, ForecastingModuleConfig
from configs.model_building_session_config import ModelBuildingSessionOutputArtifacts, ModelBuildingSessionParams, \
    ModelBuildingSessionMetrics
from general_utils import create_output_folder, render_artifact, set_dict_field, \
    get_dict_field, compute_dates
from model_wrappers.model_factory import ModelFactory
from modules.data_fetcher_module import DataFetcherModule
from modules.forecasting_module import ForecastingModule
from modules.model_evaluator import ModelEvaluator
from modules.training_module import TrainingModule
from pydantic import BaseModel
from utils.data_transformer_helper import flatten, add_init_observations_to_predictions, get_observations_subset, \
    pydantic_to_dict
from utils.io import read_file, write_file
from utils.plotting import m1_plots, m2_plots, m2_forecast_plots, distribution_plots, plot_data
from utils.time import get_date


# TODO:
# PEP8 compliance
# Exception handling - especially coming from BaseModel
# ML FLow and official pipeline setup decisions - same or separate for each city
# Message logging - currently just a simple print
# Session representation - should we choose one more level of nesting,
# e.g., data config that contains region_name, region_type, model_building_config that contains
# all items related to model_building so that the code gets simplified - only problem is that
# the referencing each nested field will require slightly longer paths.
# Proper handling to what-if-scenarios rather than the current limited hardcoding


class ModelBuildingSession(BaseModel):
    """
        Session object to encapsulate the model building
        and training process.
    """
    output_artifacts: ModelBuildingSessionOutputArtifacts
    params: ModelBuildingSessionParams
    metrics: ModelBuildingSessionMetrics

    # Constant filepaths
    _DEFAULT_SESSION_CONFIG: str = "../config/default_session_config.json"
    _ML_FLOW_CONFIG: str = "mlflow_credentials.json"
    _OFFICIAL_DATA_PIPELINE_CONFIG: str = "../../pyathena/pyathena.rc"
    _DEFAULT_ROOT_DIR: str = "../outputs/"
    _DEFAULT_MODEL_BUILDING_REPORT_TEMPLATE: str = '../src/publishers/model_building_template_v1.mustache'
    _DEFAULT_PLANNING_REPORT_TEMPLATE: str = '../src/publishers/planning_template_v1.mustache'

    def __init__(self, session_name, user_name='guest'):

        # load up the default configuration
        with open(ModelBuildingSession._DEFAULT_SESSION_CONFIG, 'r') as infile:
            super().__init__(**json.load(infile))

        # set up session_name
        self._set_session_name(session_name, user_name)

        # set up the permissions for mlflow and data pipeline
        ModelBuildingSession._init_mlflow_datapipeline_config()

        # create output folder
        create_output_folder(self.params.session_name + '_outputs', ModelBuildingSession._DEFAULT_ROOT_DIR)

    @staticmethod
    def create_session(session_name, user_name='guest'):
        # NOTE:
        # if we were going to need to inherit from this ModelBuildingSession class, we should make this as classmethod
        current_session = ModelBuildingSession(session_name, user_name)

        msg = (
            f'**********************\n'
            f'Created session: {current_session.params.session_name}\n'
            f'User: {current_session.params.user_name}\n'
            f'Output stored in the directory: {current_session.params.output_dir}\n'
            f'**********************\n'
        )
        print(msg)
        return current_session

    def _set_session_name(self, session_name, user_name):
        # create a session name if not given
        if session_name is None:
            current_date = datetime.now().date().strftime('%-m/%-d/%y')
            session_name = user_name + current_date
        self.params.session_name = session_name
        self.params.user_name = user_name

        # fix the output_directory and default paths for output artifacts
        self.params.output_dir = os.path.join(ModelBuildingSession._DEFAULT_ROOT_DIR, f'{session_name}_outputs')
        for key, value in self.output_artifacts.__fields__.items():
            self.output_artifacts.__setattr__(key, os.path.join(self.params.output_dir,
                                                                self.output_artifacts.__getattribute__(key)))

    @staticmethod
    def _init_mlflow_datapipeline_config():

        # MLflow configuration
        mlflow_config = read_file(ModelBuildingSession._ML_FLOW_CONFIG, "json", "dict")
        mlflow.set_tracking_uri(mlflow_config['tracking_url'])
        # TODO: Tracking URL as session param?
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_config['username']
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_config['password']

        # Official pipeline configuration
        os.system('source ' + ModelBuildingSession._OFFICIAL_DATA_PIPELINE_CONFIG)
        return

    def _validate_params_for_function(self, func_str):
        """
        Checks if the parameters required for a function are populated and throws an exception if not
        """
        mandatory_params = {}
        mandatory_params['view_forecasts'] = ['region_type', 'region_name']
        mandatory_params['examine_data'] = ['region_type', 'region_name']
        mandatory_params['build_models_and_generate_forecast'] = ['model_class']
        mandatory_params['generate_planning_outputs'] = []
        mandatory_params['list_outputs'] = []
        mandatory_params['log_session'] = []

        param_list = mandatory_params[func_str]

        for param in param_list:
            if not self.get_field(param):
                msg = (
                    f'Cannot run function {func_str}\n'
                    f'Mandatory input session parameters are not populated\n'
                )
                raise Exception(msg)

        return

    def get_field(self, param_name, top_field='params'):
        param_part_list = param_name.split('.')
        top_param = param_part_list[0]
        rest_params = param_part_list[1:]

        ref_obj = self.__getattribute__(top_field)
        # top level param field
        if len(param_part_list) == 1:
            output_field = ref_obj.__getattribute__(top_param)
        else:
            # nested param field (for dicts alone
            tmp_obj = ref_obj.__getattribute__(top_param)
            output_field = get_dict_field(tmp_obj, rest_params)
        return output_field

    def set_field(self, param_name, param_value, top_field='params'):
        """
        Set a specified possibly nested parameter to the specified value
        """
        # TODO: LATER Type checking
        # Does validation based on BaseModel already happen
        # What is the notion of a session? Can values once set by user again be overridden by the user?
        # There might be mismatch between logged values

        param_part_list = param_name.split('.')
        top_param = param_part_list[0]
        rest_params = param_part_list[1:]

        if top_field == 'params':
            ref_obj = self.__getattribute__(top_field)
        else:
            msg = (
                f'Unknown or uneditable session input category {top_field}\n'
                f'Cannot set field value \n'
            )
            print(msg)
            pass

        # top level param field
        if len(param_part_list) == 1:
            ref_obj.__setattr__(top_param, param_value)
        else:
            # nested param field (for dicts alone)
            tmp_obj = ref_obj.__getattribute__(top_param)
            tmp_obj = set_dict_field(tmp_obj, rest_params, param_value)
            ref_obj.__setattr__(top_param, tmp_obj)
        return

    def view_forecasts(self):
        self._validate_params_for_function('view_forecasts')
        sp = self.params
        outputs = ModelBuildingSession.view_forecasts_static(sp.experiment_name, sp.region_name,
                                                             sp.interval_to_consider)
        return outputs['links_df']

    @staticmethod
    def view_forecasts_static(experiment_name, region_name, interval_to_consider):
        outputs = {}
        outputs['links_df'] = mlflow_logger.get_previous_runs(experiment_name, region_name, interval_to_consider)
        return outputs

    def examine_data(self):
        self._validate_params_for_function('examine_data')
        sp = self.params
        outputs = ModelBuildingSession.examine_data_static(sp.region_name, sp.region_type, sp.data_source,
                                                           sp.input_data_file, self.output_artifacts)
        return

    @staticmethod
    def examine_data_static(region_name, region_type, data_source, input_data_file, output_artifacts):
        outputs = {}
        actual = DataFetcherModule.get_observations_for_region(region_type, region_name, data_source=data_source,
                                                               filepath=input_data_file, smooth=False, simple=True)

        write_file(actual, output_artifacts.cleaned_case_count_file, "csv", "dataframe")
        plot_data(region_name, actual, plot_path=output_artifacts.plot_case_count)
        return outputs

    @staticmethod
    def generate_model_operations_config(region_name, region_type, data_source, input_filepath, time_interval_config,
                                         model_class, model_parameters, train_loss_function, search_space,
                                         search_parameters, eval_loss_functions, uncertainty_parameters, model_file,
                                         operation, output_dir):

        cfg = dict()

        # set the common elements
        # Note: if the keys are aligned and this function was called as a ModelBuildingSession instance method,
        # we could have done a dict update for the subset of keys
        cfg['data_source'] = data_source
        cfg['region_name'] = region_name if isinstance(region_name, list) else [region_name]
        cfg['region_type'] = region_type
        cfg['model_class'] = model_class
        cfg['input_filepath'] = input_filepath
        # TODO: do we need this to be set really ?
        cfg['output_file_prefix'] = operation
        cfg['output_dir'] = output_dir

        # set the extra elements necessary
        if operation in ["M1_train", "M2_train"]:

            # set all the learning elements
            # TODO: make sure the module configs and the default session config are consistent
            # items to rename in the configs and also the corresponding modules
            # remove forecast_variables from forecasting module config ??
            cfg['model_parameters'] = model_parameters
            # TODO: check this Nayana - if input arg is unspecified it will be None
            cfg['model_parameters']["uncertainty_parameters"] = uncertainty_parameters
            cfg['search_space'] = search_space
            cfg['search_parameters'] = search_parameters
            cfg['train_loss_function'] = train_loss_function  # need train-config and train module change
            cfg['eval_loss_functions'] = eval_loss_functions  # need eval config and eval module change

            # set the time interval
            if operation == "M1_train":
                cfg['train_start_date'] = time_interval_config["direct"]["train1_start_date"]
                cfg['train_end_date'] = time_interval_config["direct"]["train1_end_date"]
            else:
                cfg['train_start_date'] = time_interval_config["direct"]["train2_start_date"]
                cfg['train_end_date'] = time_interval_config["direct"]["train2_end_date"]

        elif operation in ["M1_test"]:

            cfg['model_parameters'] = read_file(model_file, "json", "dict")
            cfg['eval_loss_functions'] = eval_loss_functions
            cfg['test_run_day'] = time_interval_config["direct"]["test_run_day"]
            cfg['test_start_date'] = time_interval_config["direct"]["test_start_date"]
            cfg['test_end_date'] = time_interval_config["direct"]["test_end_date"]

        # check all the different variants !
        elif operation in ['forecast']:
            # TODO: Later version - we need to set uncertainty, prediction mode, model_parameters and the dates
            cfg['model_parameters'] = read_file(model_file, "json", "dict")
            cfg['model_parameters']["uncertainty_parameters"] = uncertainty_parameters
            cfg['forecast_run_day'] = time_interval_config["direct"]["forecast_run_day"]
            cfg['forecast_start_date'] = time_interval_config["direct"]["forecast_start_date"]
            cfg['forecast_end_date'] = time_interval_config["direct"]["forecast_end_date"]
        else:
            # TODO: exception handling later
            print("Unknown model operation")
            pass

        # create the correct type of configuration
        if operation in ["M1_train", "M2_train"]:
            cfg = TrainingModuleConfig.parse_obj(cfg)
        elif operation in ["M1_test"]:
            cfg = ModelEvaluatorConfig.parse_obj(cfg)
        elif operation in ["forecast"]:
            cfg = ForecastingModuleConfig.parse_obj(cfg)
        else:
            # TODO: exception handling later
            print("Unknown model operation")
            pass

        return cfg

    def build_models_and_generate_forecast(self):
        self._validate_params_for_function('build_models_and_generate_forecast')
        sp = self.params
        outputs = ModelBuildingSession.build_models_and_generate_forecast_static(True, sp.region_name, sp.region_type,
                                                                                 sp.data_source, sp.input_data_file,
                                                                                 sp.time_interval_config,
                                                                                 sp.model_class, sp.model_parameters,
                                                                                 sp.train_loss_function,
                                                                                 sp.search_space, sp.search_parameters,
                                                                                 sp.eval_loss_functions,
                                                                                 sp.uncertainty_parameters,
                                                                                 self.output_artifacts, sp.output_dir)
        self.metrics.__fields__.update(outputs['metrics'])

        # TODO: Is there a better method of doing this?
        report_params = pydantic_to_dict(self.params)
        report_metrics = pydantic_to_dict(self.metrics)
        report_output_artifacts = pydantic_to_dict(self.output_artifacts)

        # creating report outside of the static_method to keep it simple
        reporting.create_report(report_params, report_metrics, report_output_artifacts,
                                template_path=ModelBuildingSession._DEFAULT_MODEL_BUILDING_REPORT_TEMPLATE,
                                report_path=self.output_artifacts.model_building_report)

        return outputs['metrics']

    @staticmethod
    def build_models_and_generate_forecast_static(verbose, region_name, region_type, data_source, input_filepath,
                                                  time_interval_config, model_class, model_parameters,
                                                  train_loss_function, search_space, search_parameters,
                                                  eval_loss_functions, uncertainty_parameters, output_artifacts,
                                                  output_dir):
        outputs = {}
        time_interval_config = compute_dates(time_interval_config)  # TODO: where should this go?
        metrics = ModelBuildingSession.train_eval_static(region_name, region_type, data_source, input_filepath,
                                                         time_interval_config, model_class, model_parameters,
                                                         train_loss_function, search_space, search_parameters,
                                                         eval_loss_functions, uncertainty_parameters, output_artifacts,
                                                         output_dir)

        ModelBuildingSession.forecast_and_plot_static(region_name, region_type, data_source, input_filepath,
                                                      time_interval_config, model_class, model_parameters,
                                                      uncertainty_parameters, output_artifacts, output_dir, verbose)

        outputs['metrics'] = metrics
        return outputs

    @staticmethod
    # TODO: compare if it is equivalent of the old train_eval_ensemble
    def train_eval_static(region_name, region_type, data_source, input_filepath, time_interval_config, model_class,
                          model_parameters, train_loss_function, search_space, search_parameters, eval_loss_functions,
                          uncertainty_parameters, output_artifacts, output_dir):
        metrics = {}

        # 1.  M1 train step

        # generate the config
        M1_train_config = ModelBuildingSession.generate_model_operations_config(region_name, region_type, data_source,
                                                                                input_filepath, time_interval_config,
                                                                                model_class, model_parameters,
                                                                                train_loss_function, search_space,
                                                                                search_parameters, eval_loss_functions,
                                                                                uncertainty_parameters,
                                                                                None, "M1_train", output_dir)
        # run the training
        M1_train_results = TrainingModule.from_config(deepcopy(M1_train_config))
        # collect all the outputs

        # TODO: for Harsh
        # Add beta_trials, param_ranges, percentile_params(if possible as df) in the trained ensemble model
        write_file(M1_train_results['model_parameters'], output_artifacts.M1_model_params, "json", "dict")
        # writing this out for easier access for the report TODO: ADD THIS
        # write_file(M1_train_results['model_parameters']['param_ranges'], output_artifacts.M1_param_ranges,
        #            "csv", "dataframe")
        metrics['M1_beta'] = M1_train_results['model_parameters']['beta']
        # TODO: CONVERT
        # metrics_M1_train_losses = loss_json_to_dataframe(M1_train_results['train_metric_results'], 'train1')

        # 2.  M1 test step

        # generate the config
        M1_test_config = ModelBuildingSession.generate_model_operations_config(region_name, region_type, data_source,
                                                                               input_filepath, time_interval_config,
                                                                               model_class, model_parameters,
                                                                               train_loss_function, search_space,
                                                                               search_parameters, eval_loss_functions,
                                                                               uncertainty_parameters,
                                                                               output_artifacts.M1_model_params,
                                                                               "M1_test", output_dir)
        # run the evaluation
        M1_test_results = ModelEvaluator.from_config(deepcopy(M1_test_config))

        # collect all the outputs
        # TODO: CONVERT
        # metrics_M1_test_losses = loss_json_to_dataframe(M1_test_results, 'test1')
        # metrics['M1_losses'] = pd.concat([metrics_M1_train_losses, metrics_M1_test_losses], axis=0)

        # 3. M2 train step

        # generate the config
        M2_train_config = ModelBuildingSession.generate_model_operations_config(region_name, region_type, data_source,
                                                                                input_filepath, time_interval_config,
                                                                                model_class, model_parameters,
                                                                                train_loss_function, search_space,
                                                                                search_parameters, eval_loss_functions,
                                                                                uncertainty_parameters,
                                                                                None, "M2_train", output_dir)
        # run the training
        M2_train_results = TrainingModule.from_config(deepcopy(M2_train_config))

        # collect all the outputs
        write_file(M2_train_results['model_parameters'], output_artifacts.M2_model_params, "json", "dict")
        # TODO: ADD THIS
        # write_file(M2_train_results['model_parameters']['param_ranges'], output_artifacts.M2_param_ranges,
        #            "csv", "dataframe")
        # TODO: what about percentile params - did we need to put uncertainty parameters in M2_train_config as well ?
        # If we are writing this out, would it also simplify the planning model param extraction in the planning step
        # TODO: ADD THIS
        # write_file(M2_train_results['model_parameters']['percentile_params'], output_artifacts.M2_percentile_params,
        #            "csv", "dataframe")
        metrics['M2_beta'] = M2_train_results['model_parameters']['beta']
        # TODO: CONVERT
        # metrics['M2_losses'] = loss_json_to_dataframe(M2_train_results['train_metric_results'], 'train2')

        # TODO: Make contents serializable and save
        # write_file(M1_train_config, output_artifacts.M1_train_config, "json", "dict")
        # write_file(M1_test_config, output_artifacts.M1_test_config, "json", "dict")
        # write_file(M2_train_config, output_artifacts.M2_train_config, "json", "dict")

        return metrics

    @staticmethod
    def forecast_and_plot_static(region_name, region_type, data_source, input_filepath, time_interval_config,
                                 model_class, model_parameters, uncertainty_parameters, output_artifacts, output_dir,
                                 verbose=True):
        # Get all dates of interest
        dates = time_interval_config['direct']
        forecast_end_date = dates['forecast_end_date']
        forecast_run_day = dates['forecast_run_day']
        forecast_start_date = dates['forecast_start_date']
        test_end_date = dates['test_end_date']
        test_run_day = dates['test_run_day']
        test_start_date = dates['test_start_date']
        train1_end_date = dates['train1_end_date']
        train1_run_day = dates['train1_run_day']
        train1_start_date = dates['train1_start_date']
        train2_end_date = dates['train2_end_date']
        train2_run_day = dates['train2_run_day']
        train2_start_date = dates['train2_start_date']
        plot_start_date_m1 = dates['plot_start_date_m1']
        plot_start_date_m2 = dates['plot_start_date_m2']

        # Get all the models of interest
        M1_model_params = read_file(output_artifacts.M1_model_params, 'json', 'dict')
        M2_model_params = read_file(output_artifacts.M2_model_params, 'json', 'dict')

        # Get a reusable forecast config
        # TODO: Forecasting module requires dates and model params to be set when initializing
        forecast_config = ModelBuildingSession.generate_model_operations_config(region_name, region_type, data_source,
                                                                                input_filepath, time_interval_config,
                                                                                model_class, None, None, None, None,
                                                                                None, None,
                                                                                output_artifacts.M1_model_params,
                                                                                "forecast", output_dir)

        # Get actual and smoothed observations in correct ranges
        df_m1 = DataFetcherModule.get_actual_smooth_for_region(region_type, region_name, data_source, input_filepath)
        df_m2 = DataFetcherModule.get_actual_smooth_for_region(region_type, region_name, data_source, input_filepath)

        # M1 train
        if verbose:
            df_predictions_train_m1 = ModelBuildingSession.flexible_forecast(df_m1["actual"], M1_model_params,
                                                                             train1_run_day,
                                                                             train1_start_date, forecast_end_date,
                                                                             train1_end_date, forecast_config,
                                                                             with_uncertainty=True,
                                                                             include_best_fit=True)
        else:
            df_predictions_train_m1 = ModelBuildingSession.flexible_forecast(df_m1["actual"], M1_model_params,
                                                                             train1_run_day,
                                                                             train1_start_date, train1_end_date,
                                                                             train1_end_date, forecast_config)
        # M1 test
        df_predictions_test_m1 = ModelBuildingSession.flexible_forecast(df_m1["actual"], M1_model_params, test_run_day,
                                                                        test_start_date,
                                                                        forecast_end_date,
                                                                        test_end_date, forecast_config,
                                                                        with_uncertainty=True,
                                                                        include_best_fit=True)

        # Get predictions for M2 train and forecast intervals

        # M2 train
        if verbose:
            df_predictions_train_m2 = ModelBuildingSession.flexible_forecast(df_m2["actual"], M2_model_params,
                                                                             train2_run_day,
                                                                             train2_start_date, forecast_end_date,
                                                                             train2_end_date, forecast_config,
                                                                             with_uncertainty=True,
                                                                             include_best_fit=True)
        else:
            df_predictions_train_m2 = ModelBuildingSession.flexible_forecast(df_m2["actual"], M2_model_params,
                                                                             train2_run_day,
                                                                             train2_start_date, train2_end_date,
                                                                             train2_end_date, forecast_config)

        # M2 forecast
        df_predictions_forecast_m2 = ModelBuildingSession.flexible_forecast(df_m2["actual"], M2_model_params,
                                                                            forecast_run_day,
                                                                            forecast_start_date,
                                                                            forecast_end_date, forecast_end_date,
                                                                            forecast_config, with_uncertainty=True,
                                                                            include_best_fit=True)

        # TODO: check this writing out
        write_file(df_predictions_forecast_m2, output_artifacts.M2_full_output_forecast_file, "csv", "dataframe")

        # Get percentiles to be plotted
        uncertainty_parameters = forecast_config.model_parameters['uncertainty_parameters']
        confidence_intervals = []
        for c in uncertainty_parameters['confidence_interval_sizes']:
            confidence_intervals.extend([50 - c / 2, 50 + c / 2])
        percentiles = list(set(uncertainty_parameters['percentiles'] + confidence_intervals))
        column_tags = [str(i) for i in percentiles]
        column_tags.extend(['mean', 'best'])
        region_name_str = region_name

        df_m1_plot, df_m2_plot = dict(), dict()
        df_m1_plot['actual'] = get_observations_subset(df_m1['actual'], plot_start_date_m1, test_end_date)
        df_m1_plot['smoothed'] = get_observations_subset(df_m1['smoothed'], plot_start_date_m1, test_end_date)
        df_m2_plot['actual'] = get_observations_subset(df_m2['actual'], plot_start_date_m2, train2_end_date)
        df_m2_plot['smoothed'] = get_observations_subset(df_m2['smoothed'], plot_start_date_m1, train2_end_date)

        # Create M1, M2, M2 forecast plots
        m1_plots(region_name_str, df_m1_plot["actual"], df_m1_plot["smoothed"], df_predictions_train_m1,
                 df_predictions_test_m1, train1_start_date, test_start_date, pydantic_to_dict(output_artifacts),
                 column_tags=column_tags, verbose=verbose)
        m2_plots(region_name_str, df_m2_plot["actual"], df_m2_plot["smoothed"], df_predictions_train_m2,
                 train2_start_date, pydantic_to_dict(output_artifacts),
                 column_tags=column_tags, verbose=verbose)
        m2_forecast_plots(region_name_str, df_m2_plot["actual"], df_m2_plot["smoothed"], df_predictions_forecast_m2,
                          train2_start_date, forecast_start_date, pydantic_to_dict(output_artifacts),
                          column_tags=column_tags, verbose=False)

        # Get trials dataframe
        region_metadata = DataFetcherModule.get_regional_metadata(region_type, region_name, data_source=data_source)
        M2_model = ModelFactory.get_model(model_class, M2_model_params)
        # TODO: LATER - Round2 refactoring
        trials = M2_model.get_trials_distribution(region_metadata, df_m2["actual"], forecast_run_day,
                                                  forecast_start_date, forecast_end_date)
        # Plot PDF and CDF
        distribution_plots(trials, uncertainty_parameters['variable_of_interest'], pydantic_to_dict(output_artifacts))

    @staticmethod
    def flexible_forecast(actual, model_params, forecast_run_day, forecast_start_date, forecast_end_date,
                          forecast_trim_day, forecast_config, with_uncertainty=False, include_best_fit=False):

        # forecast_config = ForecastingModuleConfig.parse_obj(forecast_config)

        # set the dates and the model parameters
        forecast_config.model_parameters = model_params
        forecast_config.forecast_run_day = forecast_run_day
        forecast_config.forecast_start_date = forecast_start_date
        forecast_config.forecast_end_date = forecast_end_date

        # change the predict mode
        if with_uncertainty:
            forecast_config.model_parameters['modes']['predict_mode'] = 'predictions_with_uncertainty'

        forecast_df = ForecastingModule.from_config(forecast_config)
        forecast_df_best_fit = pd.DataFrame()
        if include_best_fit:
            forecast_config.model_parameters['modes']['predict_mode'] = 'best_fit'
            forecast_df_best_fit = ForecastingModule.from_config(forecast_config)
            forecast_df_best_fit = forecast_df_best_fit.drop(
                columns=['Region Type', 'Region', 'Country', 'Lat', 'Long'])
            for col in forecast_df_best_fit.columns:
                if col.endswith('_mean'):
                    new_col = '_'.join([col.split('_')[0], 'best'])
                    forecast_df_best_fit = forecast_df_best_fit.rename(columns={col: new_col})
                else:
                    forecast_df_best_fit = forecast_df_best_fit.rename(columns={col: '_'.join([col, 'best'])})

        forecast_df = forecast_df.drop(columns=['Region Type', 'Region', 'Country', 'Lat', 'Long'])
        forecast_df = pd.concat([forecast_df_best_fit, forecast_df], axis=1)
        forecast_df = forecast_df.reset_index()

        # add run day observation and trim
        forecast_df = add_init_observations_to_predictions(actual, forecast_df,
                                                           forecast_run_day)
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        forecast_df = forecast_df[forecast_df['date'] < get_date(forecast_trim_day, 1)]
        return forecast_df

    def generate_planning_outputs(self):
        self._validate_params_for_function('generate_planning_outputs')
        sp = self.params

        # NOTE: we kept the model_params transfer explicit and separate from that of output_artifacts in case we
        # we change how we log and transfer objects during the session
        outputs = ModelBuildingSession.generate_planning_outputs_static(sp.region_name, sp.region_type, sp.data_source,
                                                                        sp.input_data_file, sp.time_interval_config,
                                                                        sp.model_class,
                                                                        sp.uncertainty_parameters, sp.planning,
                                                                        sp.staffing,
                                                                        self.output_artifacts.M2_model_params,
                                                                        self.output_artifacts, sp.output_dir)
        self.metrics.__fields__.update(outputs['metrics'])

        # TODO: Is there a better method of doing this?
        report_params = pydantic_to_dict(self.params)
        report_metrics = pydantic_to_dict(self.metrics)
        report_output_artifacts = pydantic_to_dict(self.output_artifacts)

        # creating report outside of the static_method to keep it simple
        reporting.create_report(report_params, report_metrics, report_output_artifacts,
                                template_path=ModelBuildingSession._DEFAULT_PLANNING_REPORT_TEMPLATE,
                                report_path=self.output_artifacts.planning_report)

        return outputs['metrics']

    @staticmethod
    def generate_planning_outputs_static(region_name, region_type, data_source, input_data_file,
                                         time_interval_config, model_class, uncertainty_parameters, planning,
                                         staffing, model_file, output_artifacts, output_dir):
        outputs = {}

        # 1. read the ensemble model file
        M2_model_params = read_file(model_file, "json", "dict")
        M2_model = ModelFactory.get_model(model_class, M2_model_params)

        dates = compute_dates(time_interval_config)['direct']

        # 2. get the relevant data
        regional_data = DataFetcherModule.get_regional_data(region_type, [region_name], data_source=data_source,
                                                            input_filepath=input_data_file)

        # 3. get the representative model for the chosen planning level
        # TODO: Is this always a single percentile?
        planning_percentile = planning['ref_level'] if isinstance(planning['ref_level'], list) else [
            planning['ref_level']]
        percentile_params = M2_model.get_params_for_percentiles(
            variable_of_interest=uncertainty_parameters['variable_of_interest'],
            date_of_interest=uncertainty_parameters['date_of_interest'],
            tolerance=uncertainty_parameters['tolerance'], percentiles=planning_percentile,
            region_metadata=regional_data['metadata'], region_observations=regional_data['actual'],
            run_day=dates['forecast_run_day'], start_date=dates['forecast_start_date'],
            end_date=dates['forecast_end_date'])

        M2_planning_model_params = percentile_params[planning['ref_level']]
        write_file(M2_planning_model_params, output_artifacts.M2_planning_model_params, "json", "dict")

        # 4. generate the required forecast config for the child models
        # TODO: check how we could get this without hardcoding it
        child_model_class = "SEIHRD_gen"
        # TODO: Check if this is correct
        m2_child_forecast_config = ModelBuildingSession.generate_model_operations_config(
            region_name, region_type, data_source, input_data_file, time_interval_config, child_model_class,
            None, None, None, None, None, None, output_artifacts.M2_planning_model_params, "forecast", output_dir)

        # 5. compute the parameters (just r0) for the different scenarios including the planning one
        rt_multiplier_list = [1]
        rt_multiplier_list.extend(planning['rt_multiplier_list'])
        rt_list = [r * M2_planning_model_params['r0'] for r in rt_multiplier_list]
        metrics = {"M2_scenarios_r0": rt_list}

        # 6. loop through the different cases
        # TODO: LATER V3 generalize this piece for handling more complex scenarios - later V3
        for i in range(len(rt_list)):

            #  choose the names for each case - for artifact files
            if i == 0:
                case_name = "planning"
            else:
                case_name = "scenario_" + str(i)

            #  set the model params correctly
            m2_child_forecast_config.model_parameters["r0"] = rt_list[i]

            #  compute and save the case count forecast
            # TODO: Need to change saving all the planning and scenario forecasts into a single csv
            # Right now code it will break unless we add extra forecast csv artifacts for other scenarios
            forecasting_output = ForecastingModule.from_config(m2_child_forecast_config)
            forecast_file_key = f'staffing_{case_name}'
            forecasting_output.to_csv(output_artifacts.__getattribute__(forecast_file_key), index=False)

            # compute and save the staffing matrices
            active_count = float(
                forecasting_output[forecasting_output.index == uncertainty_parameters['date_of_interest']][
                    'active_mean'])
            staffing_df = domain_info.compute_staffing_matrix(active_count, staffing['bed_type_ratio'],
                                                              staffing['staffing_ratios_file'],
                                                              staffing['bed_multiplier_count'])
            staff_matrix_file_key = "staffing_" + case_name
            staffing_df.to_csv(output_artifacts.__getattribute__(staff_matrix_file_key), index=False)

            # generate and save the plots (only CARDs)
            df = ModelBuildingSession._generate_forecast_plot_data(region_type, region_name, data_source,
                                                                   input_data_file, forecasting_output, dates)
            # Is there an option in the function for only CARD plots
            m2_forecast_plots(region_name, df["actual_m2"], df["smoothed_m2"], df["predictions_forecast_m2"],
                              dates["train2_start_date"], dates["forecast_start_date"],
                              pydantic_to_dict(output_artifacts), column_tags=['mean'], verbose=False,
                              scenario=case_name)

        outputs['metrics'] = metrics
        return outputs

    def list_outputs(self):
        self._validate_params_for_function('list_outputs')
        outputs = ModelBuildingSession.list_outputs_static(self.output_artifacts)
        return

    @staticmethod
    def list_outputs_static(output_artifacts, display_list=['model_building_report', 'planning_report']):
        """

        Args:
            output_artifacts ():
            display_list ():

        Returns:

        """
        outputs = {}
        print('Listing all the output artifacts from the session:\n')
        for key, value in output_artifacts.__fields__.items():
            print(key + ': ' + value.name + '\n')

        for a in display_list:
            render_artifact(output_artifacts.__getattribute__(a), ".md")
        return outputs

    def log_session(self):
        self._validate_params_for_function('log_session')
        # Todo: Is this necessary? Can we flatten a pydantic object before logging?
        params_to_log = pydantic_to_dict(self.params)
        metrics_to_log = pydantic_to_dict(self.metrics)
        output_artifacts_to_log = pydantic_to_dict(self.output_artifacts)
        outputs = ModelBuildingSession.log_session_static(params_to_log, metrics_to_log, output_artifacts_to_log,
                                                          self.params.experiment_name, self.params.session_name)
        return

    @staticmethod
    def log_session_static(params, metrics, output_artifacts, experiment_name, session_name):
        # Todo: Check for non-numeric types for metrics
        outputs = {'flattened_params': flatten(params), 'flattened_metrics': flatten(metrics)}
        # TODO: Locate input artifacts in params? Or specify full path in session_config and use get_field?
        # input_artifacts = {(k, params[k]) for k in params['input_artifacts']}
        # outputs['flattened_artifacts'] = merge_dicts([flatten(input_artifacts), flatten(output_artifacts)])
        outputs['flattened_artifacts'] = flatten(output_artifacts)
        mlflow_logger.log_to_mlflow(outputs['flattened_params'], outputs['flattened_metrics'],
                                    outputs['flattened_artifacts'], experiment_name, session_name)
        return outputs

    # TODO: not a clean function but just put this together to clean up the main calls
    # Need to decide where this goes as well
    @staticmethod
    def _generate_forecast_plot_data(region_type, region_name, data_source, input_data_file, forecasting_output,
                                     time_interval_config):
        # TODO: check do we need this ?
        forecasting_output = forecasting_output.reset_index()

        df = DataFetcherModule.get_actual_smooth_for_region(region_type, [region_name], data_source=data_source,
                                                            input_filepath=input_data_file)

        df["actual_m2"] = get_observations_subset(df["actual"], time_interval_config['plot_start_date_m2'],
                                                  time_interval_config["train2_end_date"])
        df["smoothed_m2"] = get_observations_subset(df["smoothed"], time_interval_config['plot_start_date_m2'],
                                                    time_interval_config["train2_end_date"])

        df["predictions_forecast_m2"] = add_init_observations_to_predictions(df["actual"], forecasting_output,
                                                                             time_interval_config["forecast_run_day"])
        df["predictions_forecast_m2"]['date'] = pd.to_datetime(df["predictions_forecast_m2"]['date'], format='%m/%d/%y')

        return df
