import os
import logging
from copy import deepcopy
from datetime import datetime

import domain_info.staffing as domain_info
import mlflow
import pandas as pd
import publishers.mlflow_logging as mlflow_logger
import publishers.report_generation as reporting
from configs.base_config import TrainingModuleConfig, ModelEvaluatorConfig, ForecastingModuleConfig
from configs.model_building_session_config import ModelBuildingSessionOutputArtifacts, ModelBuildingSessionParams, \
    ModelBuildingSessionMetrics
from model_wrappers.model_factory import ModelFactory
from modules.data_fetcher_module import DataFetcherModule
from modules.forecasting_module import ForecastingModule
from modules.model_evaluator import ModelEvaluator
from modules.training_module import TrainingModule
from pydantic import BaseModel
from utils.data_util import get_observations_subset, to_dict, get_date
from utils.general_utils import create_output_folder, render_artifact, set_dict_field, get_dict_field
from utils.io import read_file, write_file
from utils.metrics_util import loss_to_dataframe
from utils.plotting import m1_plots, m2_plots, m2_forecast_plots, distribution_plots, multivariate_case_count_plot


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
    _ML_FLOW_CONFIG: str = "../.keys/mlflow_credentials.json"
    _OFFICIAL_DATA_PIPELINE_CONFIG: str = "../.keys/pyathena.rc"
    _DEFAULT_ROOT_DIR: str = "../outputs/"
    _DEFAULT_MODEL_BUILDING_REPORT_TEMPLATE: str = '../src/publishers/model_building_template_v1.mustache'
    _DEFAULT_PLANNING_REPORT_TEMPLATE: str = '../src/publishers/planning_template_v1.mustache'

    def __init__(self, session_name, user_name='guest'):

        # load up the default configuration
        super().__init__(**read_file(ModelBuildingSession._DEFAULT_SESSION_CONFIG, "json", "dict"))

        # set up session_name
        self._set_session_name(session_name, user_name)

        # set up the permissions for mlflow and data pipeline
        ModelBuildingSession._init_mlflow_datapipeline_config()

        # create output folder
        create_output_folder(self.params.session_name + '_outputs', ModelBuildingSession._DEFAULT_ROOT_DIR)

        # set up logging
        logging.basicConfig(filename=self.output_artifacts.session_log)

    def _set_session_name(self, session_name, user_name):
        """

        Args:
            session_name (str): Session name variable
            user_name (str): Name of the user running the experiment

        Returns:

        """
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

    def _validate_params_for_function(self, func_str):
        """Checks if the parameters required for a function are populated and throws an exception if not

        Args:
            func_str (str): 

        Returns:

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
        """

        Args:
            param_name (str): 
            top_field (str, optional): 

        Returns:

        """
        param_part_list = param_name.split('.')
        top_param = param_part_list[0]
        rest_params = param_part_list[1:]

        ref_obj = self.__getattribute__(top_field)
        # top level param field
        if len(param_part_list) == 1:
            output_field = ref_obj.__getattribute__(top_param)
        else:
            # nested param field (for dicts alone)
            tmp_obj = ref_obj.__getattribute__(top_param)
            output_field = get_dict_field(tmp_obj, rest_params)
        return output_field

    def set_field(self, param_name, param_value, top_field='params'):
        """Set a specified possibly nested parameter to the specified value

        Args:
            param_name (str): 
            param_value (Any): 
            top_field (str): 

        Returns:

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
        """

        Returns:

        """
        self._validate_params_for_function('view_forecasts')
        sp = self.params
        outputs = ModelBuildingSession.view_forecasts_static(sp.experiment_name, sp.region_name,
                                                             sp.interval_to_consider)
        return outputs['links_df']

    def examine_data(self):
        """

        Returns:

        """
        self._validate_params_for_function('examine_data')
        sp = self.params
        outputs = ModelBuildingSession.examine_data_static(sp.region_name, sp.region_type, sp.data_source,
                                                           sp.input_file_path, self.output_artifacts)
        return

    def build_models_and_generate_forecast(self):
        """

        Returns:

        """
        self._validate_params_for_function('build_models_and_generate_forecast')
        sp = self.params
        sp.time_interval_config = ModelBuildingSession._compute_dates(sp.time_interval_config)
        outputs = ModelBuildingSession.build_models_and_generate_forecast_static(sp.region_name, sp.region_type,
                                                                                 sp.data_source, sp.input_file_path,
                                                                                 sp.time_interval_config,
                                                                                 sp.model_class, sp.model_parameters,
                                                                                 sp.train_loss_function,
                                                                                 sp.search_space, sp.search_parameters,
                                                                                 sp.eval_loss_functions,
                                                                                 sp.uncertainty_parameters,
                                                                                 self.output_artifacts, sp.output_dir,
                                                                                 True)

        for metric, value in outputs['metrics'].items():
            self.metrics.__setattr__(metric, value)

        # creating report outside of the static_method to keep it simple
        reporting.create_report(self.params, self.metrics, self.output_artifacts,
                                artifacts_to_render=self.params.artifacts_to_render,
                                template_path=ModelBuildingSession._DEFAULT_MODEL_BUILDING_REPORT_TEMPLATE,
                                report_path=self.output_artifacts.model_building_report)

        return outputs['metrics']

    def generate_planning_outputs(self):
        """

        Returns:

        """
        self._validate_params_for_function('generate_planning_outputs')
        sp = self.params

        # NOTE: we kept the model_params transfer explicit and separate from that of output_artifacts in case we
        # we change how we log and transfer objects during the session
        outputs = ModelBuildingSession.generate_planning_outputs_static(sp.region_name, sp.region_type, sp.data_source,
                                                                        sp.input_file_path, sp.time_interval_config,
                                                                        sp.model_class,
                                                                        sp.uncertainty_parameters, sp.planning,
                                                                        sp.staffing,
                                                                        self.output_artifacts.M2_model_params,
                                                                        self.output_artifacts, sp.output_dir)
        self.metrics.__fields__.update(outputs['metrics'])

        # creating report outside of the static_method to keep it simple
        reporting.create_report(self.params, self.metrics, self.output_artifacts,
                                template_path=ModelBuildingSession._DEFAULT_PLANNING_REPORT_TEMPLATE,
                                report_path=self.output_artifacts.planning_report)

        return outputs['metrics']

    def list_outputs(self):
        """

        Returns:

        """
        self._validate_params_for_function('list_outputs')
        outputs = ModelBuildingSession.list_outputs_static(self.output_artifacts)
        return

    def log_session(self):
        """

        Returns:

        """
        self._validate_params_for_function('log_session')
        outputs = ModelBuildingSession.log_session_static(self.params, self.metrics, self.output_artifacts,
                                                          self.params.experiment_name, self.params.session_name)
        return

    @staticmethod
    def _init_mlflow_datapipeline_config():
        """

        Returns:

        """
        # MLflow configuration
        try:
            mlflow_config = read_file(ModelBuildingSession._ML_FLOW_CONFIG, "json", "dict")
            mlflow.set_tracking_uri(mlflow_config['tracking_url'])
            os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_config['username']
            os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_config['password']
        except FileNotFoundError:
            print('MLflow configuration not found')

        # Official pipeline configuration
        if os.system('source ' + ModelBuildingSession._OFFICIAL_DATA_PIPELINE_CONFIG) != 0:
            print('Official data pipeline not setup')

    @staticmethod
    def _compute_dates(time_interval_config):
        """Compute all the dates in time_interval_config in a consistent way

        Args:
            time_interval_config (dict): Dict comprising all the key dates or alternately interval durations for training, evaluation, forecast and plotting. See Session_Definition for more info.

        Returns:

        """
        # Note: expectation is that either the user specifies ALL the dates in "direct" mode
        # or we would populate it based on the offsets and reference day
        is_direct_mode = True
        for k in time_interval_config["direct"]:
            if not time_interval_config["direct"][k]:
                is_direct_mode = False
                break

        # if not direct/expert mode, reference_day is set to today if not specified by user
        # and other dates in the direct mode are computed accordingly
        if not is_direct_mode:

            if not time_interval_config["offset_based"]["reference_day"]:
                d0 = datetime.today().strftime("%-m/%-d/%y")
                time_interval_config["offset_based"]["reference_day"] = d0
            else:  # Added
                d0 = time_interval_config["offset_based"]["reference_day"]
            # compute the direct_mode dates based on this
            d = {}
            offsets = time_interval_config["offset_based"]
            d["forecast_end_date"] = get_date(d0, offsets["forecast_period"])
            d["forecast_run_day"] = get_date(d0, 0)
            d["forecast_start_date"] = get_date(d0, 1)
            d["test_end_date"] = get_date(d0, 0)
            d["test_run_day"] = get_date(d0, -offsets["test_period"] - 1)
            d["test_start_date"] = get_date(d0, -offsets["test_period"])
            d["train1_end_date"] = get_date(d0, -offsets["test_period"] - 1)
            d["train1_run_day"] = get_date(d0, -offsets["test_period"] - offsets["train_period"] - 2)
            d["train1_start_date"] = get_date(d0, -offsets["test_period"] - offsets["train_period"] - 1)
            d["train2_end_date"] = get_date(d0, 0)
            d["train2_run_day"] = get_date(d0, -offsets["train_period"] - 1)
            d["train2_start_date"] = get_date(d0, -offsets["train_period"])
            d["plot_start_date_m1"] = get_date(d0, -offsets["test_period"] - offsets["train_period"] - offsets[
                "plot_buffer"] - 1)
            d["plot_start_date_m2"] = get_date(d0, -offsets["train_period"] - offsets["plot_buffer"])
            time_interval_config["direct"] = d

        return time_interval_config

    @staticmethod
    def _generate_model_operations_config(region_name, region_type, data_source, input_file_path, time_interval_config,
                                          model_class, model_parameters, train_loss_function, search_space,
                                          search_parameters, eval_loss_functions, uncertainty_parameters, model_file,
                                          operation, output_dir):
        """

        Args:
            region_name (list): Region of interest (list of regions possibly singleton) for building the forecasting models
            region_type (str): Type of region [district, state]
            data_source (DataSource): Source of input data [direct_csv, official_data, rootnet_stats_history, tracker_data_all] (default tracker_data_all)
            input_file_path (str): Path to the input csv if data_source is set to direct_csv
            time_interval_config (dict): Dict comprising all the key dates or alternately interval durations for training, evaluation, forecast and plotting. See Session_Definition for more info.
            model_class (ModelClass): Model class of the ensemble model [homogeneous_ensemble, heterogeneous_ensemble]
            model_parameters (dict, optional): Dict comprising model parameters. Pre-training, only a subset of choices are specified and post training, it is fully populated. See Session_Definition and Modelling_Details for more info.
            train_loss_function (dict, optional): Dict to specify the loss function (loss metric and variable weights) to be used for training.
            search_space (dict, optional): Dict that specifies the search space (lower and upper bounds) of all the model parameters.
            search_parameters (dict, optional): Dict that specified parameters of the optimization process such as number of trials. See Session_Definition for more info.
            eval_loss_functions (list, optional):  List of loss functions (dicts) for specifying the metric_name (default: mape) and weight on the compartment (default : 1)
            uncertainty_parameters (dict, optional): Dict comprising parameters such as confidence_interval, percentiles required for generating a forecast distribution. See Session_Definition for more info.
            model_file (str, optional): 
            operation (str): 
            output_dir (str): Path to the output directory for saving artifacts, trained models, forecasts, plots and reports

        Returns:

        """

        cfg = dict()

        # set the common elements
        # Note: if the keys are aligned and this function was called as a ModelBuildingSession instance method,
        # we could have done a dict update for the subset of keys
        cfg['data_source'] = data_source
        cfg['region_name'] = region_name if isinstance(region_name, list) else [region_name]
        cfg['region_type'] = region_type
        cfg['model_class'] = model_class
        cfg['input_filepath'] = input_file_path
        cfg['output_file_prefix'] = operation
        cfg['output_dir'] = output_dir

        # set the extra elements necessary
        if operation in ["M1_train", "M2_train"]:

            # set all the learning elements
            # TODO: make sure the module configs and the default session config are consistent
            # items to rename in the configs and also the corresponding modules
            # remove forecast_variables from forecasting module config ??
            cfg['model_parameters'] = model_parameters
            # TODO: if input arg is unspecified it will be None
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

    @staticmethod
    def create_session(session_name, user_name='guest'):
        """

        Args:
            session_name (str): Session name variable
            user_name (str): Name of the user running the experiment

        Returns:

        """
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

    @staticmethod
    def view_forecasts_static(experiment_name, region_name, interval_to_consider):
        """

        Args:
            experiment_name (str): MLFlow experiment name
            region_name (str): Region of interest (list of regions possibly singleton) for building the forecasting models
            interval_to_consider (int): Number of days prior to the current date to start the search

        Returns:

        """
        outputs = {}
        try:
            outputs['links_df'] = mlflow_logger.get_previous_runs(experiment_name, region_name, interval_to_consider)
        except:
            pass
        return outputs

    @staticmethod
    def examine_data_static(region_name, region_type, data_source, input_file_path, output_artifacts):
        """

        Args:
            region_name (list): Region of interest (list of regions possibly singleton) for building the forecasting models
            region_type (str): Type of region [district, state]
            data_source (DataSource): Source of input data [direct_csv, official_data, rootnet_stats_history, tracker_data_all] (default tracker_data_all)
            input_file_path (str): Path to the input csv if data_source is set to direct_csv
            output_artifacts (ModelBuildingSessionOutputArtifacts): List of artifacts generated in the model building session (model jsons, pngs, csvs)

        Returns:

        """
        outputs = {}
        actual = DataFetcherModule.get_observations_for_region(region_type, region_name, data_source=data_source,
                                                               filepath=input_file_path, smooth=False, simple=True)
        write_file(actual, output_artifacts.cleaned_case_count_file, "csv", "dataframe")
        multivariate_case_count_plot(actual, column_label='', column_tag='', title=region_name,
                                     path=output_artifacts.plot_case_count)
        return outputs

    @staticmethod
    def build_models_and_generate_forecast_static(region_name, region_type, data_source, input_file_path,
                                                  time_interval_config, model_class, model_parameters,
                                                  train_loss_function, search_space, search_parameters,
                                                  eval_loss_functions, uncertainty_parameters, output_artifacts,
                                                  output_dir, verbose):
        """

        Args:
            verbose (bool): 
            region_name (list): Region of interest (list of regions possibly singleton) for building the forecasting models
            region_type (str): Type of region [district, state]
            data_source (DataSource): Source of input data [direct_csv, official_data, rootnet_stats_history, tracker_data_all] (default tracker_data_all)
            input_file_path (str): Path to the input csv if data_source is set to direct_csv
            time_interval_config (dict): Dict comprising all the key dates or alternately interval durations for training, evaluation, forecast and plotting. See Session_Definition for more info.
            model_class (str): Model class of the ensemble model [homogeneous_ensemble, heterogeneous_ensemble]
            model_parameters (dict): Dict comprising model parameters. Pre-training, only a subset of choices are specified and post training, it is fully populated. See Session_Definition and Modelling_Details for more info.
            train_loss_function (dict): Dict to specify the loss function (loss metric and variable weights) to be used for training.
            search_space (dict): Dict that specifies the search space (lower and upper bounds) of all the model parameters.
            search_parameters (dict): Dict that specified parameters of the optimization process such as number of trials. See Session_Definition for more info.
            eval_loss_functions (list):  List of loss functions (dicts) for specifying the metric_name (default: mape) and weight on the compartment (default : 1)
            uncertainty_parameters (dict): Dict comprising parameters such as confidence_interval, percentiles required for generating a forecast distribution. See Session_Definition for more info.
            output_artifacts (ModelBuildingSessionOutputArtifacts): List of artifacts generated in the model building session (model jsons, pngs, csvs)
            output_dir (str): Path to the output directory for saving artifacts, trained models, forecasts, plots and reports

        Returns:

        """
        outputs = {}
        metrics = ModelBuildingSession.train_eval_static(region_name, region_type, data_source, input_file_path,
                                                         time_interval_config, model_class, model_parameters,
                                                         train_loss_function, search_space, search_parameters,
                                                         eval_loss_functions, uncertainty_parameters, output_artifacts,
                                                         output_dir)

        ModelBuildingSession.forecast_and_plot_static(region_name, region_type, data_source, input_file_path,
                                                      time_interval_config, model_class, model_parameters,
                                                      uncertainty_parameters, output_artifacts, output_dir, verbose)

        outputs['metrics'] = metrics
        return outputs

    @staticmethod
    def train_eval_static(region_name, region_type, data_source, input_file_path, time_interval_config, model_class,
                          model_parameters, train_loss_function, search_space, search_parameters, eval_loss_functions,
                          uncertainty_parameters, output_artifacts, output_dir):
        """

        Args:
            region_name (list): Region of interest (list of regions possibly singleton) for building the forecasting models
            region_type (str): Type of region [district, state]
            data_source (DataSource): Source of input data [direct_csv, official_data, rootnet_stats_history, tracker_data_all] (default tracker_data_all)
            input_file_path (str): Path to the input csv if data_source is set to direct_csv
            time_interval_config (dict): Dict comprising all the key dates or alternately interval durations for training, evaluation, forecast and plotting. See Session_Definition for more info.
            model_class (str): Model class of the ensemble model [homogeneous_ensemble, heterogeneous_ensemble]
            model_parameters (dict): Dict comprising model parameters. Pre-training, only a subset of choices are specified and post training, it is fully populated. See Session_Definition and Modelling_Details for more info.
            train_loss_function (dict): Dict to specify the loss function (loss metric and variable weights) to be used for training.
            search_space (dict): Dict that specifies the search space (lower and upper bounds) of all the model parameters.
            search_parameters (dict): Dict that specified parameters of the optimization process such as number of trials. See Session_Definition for more info.
            eval_loss_functions (list):  List of loss functions (dicts) for specifying the metric_name (default: mape) and weight on the compartment (default : 1)
            uncertainty_parameters (dict): Dict comprising parameters such as confidence_interval, percentiles required for generating a forecast distribution. See Session_Definition for more info.
            output_artifacts (ModelBuildingSessionOutputArtifacts): List of artifacts generated in the model building session (model jsons, pngs, csvs)
            output_dir (str): Path to the output directory for saving artifacts, trained models, forecasts, plots and reports

        Returns:

        """
        metrics = {}

        # 1.  M1 train step

        # generate the config
        M1_train_config = ModelBuildingSession._generate_model_operations_config(region_name, region_type, data_source,
                                                                                 input_file_path, time_interval_config,
                                                                                 model_class, model_parameters,
                                                                                 train_loss_function, search_space,
                                                                                 search_parameters, eval_loss_functions,
                                                                                 uncertainty_parameters,
                                                                                 None, "M1_train", output_dir)
        # run the training
        print('Performing M1 training...')
        M1_train_results = TrainingModule.from_config(deepcopy(M1_train_config))
        # collect all the outputs

        # Save beta_trials, param_ranges, percentile_params, model_params, metric results
        write_file(to_dict(M1_train_results['model']), output_artifacts.M1_model, "json", "dict")
        write_file(M1_train_results['model_parameters'], output_artifacts.M1_model_params, "json", "dict")
        write_file(M1_train_results['trials'], output_artifacts.M1_beta_trials, "json", "dict")
        write_file(M1_train_results['param_ranges'], output_artifacts.M1_param_ranges, "csv", "dataframe")
        metrics['M1_beta'] = M1_train_results['model_parameters']['beta']
        metrics_M1_train_losses = loss_to_dataframe(M1_train_results['train_metric_results'], 'train1')

        # 2.  M1 test step

        # generate the config
        M1_test_config = ModelBuildingSession._generate_model_operations_config(region_name, region_type, data_source,
                                                                                input_file_path, time_interval_config,
                                                                                model_class, model_parameters,
                                                                                train_loss_function, search_space,
                                                                                search_parameters, eval_loss_functions,
                                                                                uncertainty_parameters,
                                                                                output_artifacts.M1_model_params,
                                                                                "M1_test", output_dir)
        # run the evaluation
        M1_test_results = ModelEvaluator.from_config(deepcopy(M1_test_config))

        # collect all the outputs
        metrics_M1_test_losses = loss_to_dataframe(M1_test_results, 'test1')
        metrics['M1_losses'] = pd.concat([metrics_M1_train_losses, metrics_M1_test_losses], axis=0)

        # 3. M2 train step

        # generate the config
        print('Performing M2 training...')
        M2_train_config = ModelBuildingSession._generate_model_operations_config(region_name, region_type, data_source,
                                                                                 input_file_path, time_interval_config,
                                                                                 model_class, model_parameters,
                                                                                 train_loss_function, search_space,
                                                                                 search_parameters, eval_loss_functions,
                                                                                 uncertainty_parameters,
                                                                                 None, "M2_train", output_dir)
        # run the training
        M2_train_results = TrainingModule.from_config(deepcopy(M2_train_config))

        # TODO: what about percentile params - did we need to put uncertainty parameters in M2_train_config as well ?
        # collect all the outputs
        write_file(to_dict(M2_train_results['model']), output_artifacts.M2_model, "json", "dict")
        write_file(M2_train_results['model_parameters'], output_artifacts.M2_model_params, "json", "dict")
        write_file(M2_train_results['trials'], output_artifacts.M2_beta_trials, "json", "dict")
        write_file(M2_train_results['param_ranges'], output_artifacts.M2_param_ranges, "csv", "dataframe")
        write_file(M2_train_results['percentile_parameters'], output_artifacts.M2_percentile_params, "csv", "dataframe")
        metrics['M2_beta'] = M2_train_results['model_parameters']['beta']
        metrics['M2_losses'] = loss_to_dataframe(M2_train_results['train_metric_results'], 'train2')

        # write out configs
        write_file(to_dict(M1_train_config), output_artifacts.M1_train_config, "json", "dict", indent=4)
        write_file(to_dict(M1_test_config), output_artifacts.M1_test_config, "json", "dict", indent=4)
        write_file(to_dict(M2_train_config), output_artifacts.M2_train_config, "json", "dict", indent=4)

        return metrics

    @staticmethod
    def forecast_and_plot_static(region_name, region_type, data_source, input_file_path, time_interval_config,
                                 model_class, model_parameters, uncertainty_parameters, output_artifacts, output_dir,
                                 verbose=True):
        """

        Args:
            region_name (list): Region of interest (list of regions possibly singleton) for building the forecasting models
            region_type (str): Type of region [district, state]
            data_source (DataSource): Source of input data [direct_csv, official_data, rootnet_stats_history, tracker_data_all] (default tracker_data_all)
            input_file_path (str): Path to the input csv if data_source is set to direct_csv
            time_interval_config (dict): Dict comprising all the key dates or alternately interval durations for training, evaluation, forecast and plotting. See Session_Definition for more info.
            model_class (ModelClass): Model class of the ensemble model [homogeneous_ensemble, heterogeneous_ensemble]
            model_parameters (dict): Dict comprising model parameters. Pre-training, only a subset of choices are specified and post training, it is fully populated. See Session_Definition and Modelling_Details for more info.
            uncertainty_parameters (dict): Dict comprising parameters such as confidence_interval, percentiles required for generating a forecast distribution. See Session_Definition for more info.
            output_artifacts (ModelBuildingSessionOutputArtifacts): List of artifacts generated in the model building session (model jsons, pngs, csvs)
            output_dir (str): Path to the output directory for saving artifacts, trained models, forecasts, plots and reports
            verbose (bool): 

        Returns:

        """
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
        forecast_config = ModelBuildingSession._generate_model_operations_config(region_name, region_type, data_source,
                                                                                 input_file_path, time_interval_config,
                                                                                 model_class, None, None, None, None,
                                                                                 None, None,
                                                                                 output_artifacts.M1_model_params,
                                                                                 "forecast", output_dir)

        # Get actual and smoothed observations in correct ranges
        df_m1 = DataFetcherModule.get_actual_smooth_for_region(region_type, region_name, data_source, input_file_path)
        df_m2 = DataFetcherModule.get_actual_smooth_for_region(region_type, region_name, data_source, input_file_path)

        print('Getting M1 forecasts...')
        # M1 train
        if verbose:
            df_predictions_train_m1 = ForecastingModule.flexible_forecast(df_m1["actual"], M1_model_params,
                                                                          train1_run_day,
                                                                          train1_start_date, forecast_end_date,
                                                                          train1_end_date, forecast_config,
                                                                          with_uncertainty=True,
                                                                          include_best_fit=True)
        else:
            df_predictions_train_m1 = ForecastingModule.flexible_forecast(df_m1["actual"], M1_model_params,
                                                                          train1_run_day,
                                                                          train1_start_date, train1_end_date,
                                                                          train1_end_date, forecast_config)
        # M1 test
        df_predictions_test_m1 = ForecastingModule.flexible_forecast(df_m1["actual"], M1_model_params, test_run_day,
                                                                     test_start_date,
                                                                     forecast_end_date,
                                                                     test_end_date, forecast_config,
                                                                     with_uncertainty=True,
                                                                     include_best_fit=True)

        # Get predictions for M2 train and forecast intervals
        print('Getting M2 forecasts...')
        # M2 train
        if verbose:
            df_predictions_train_m2 = ForecastingModule.flexible_forecast(df_m2["actual"], M2_model_params,
                                                                          train2_run_day,
                                                                          train2_start_date, forecast_end_date,
                                                                          train2_end_date, forecast_config,
                                                                          with_uncertainty=True,
                                                                          include_best_fit=True)
        else:
            df_predictions_train_m2 = ForecastingModule.flexible_forecast(df_m2["actual"], M2_model_params,
                                                                          train2_run_day,
                                                                          train2_start_date, train2_end_date,
                                                                          train2_end_date, forecast_config)

        # M2 forecast
        df_predictions_forecast_m2 = ForecastingModule.flexible_forecast(df_m2["actual"], M2_model_params,
                                                                         forecast_run_day,
                                                                         forecast_start_date,
                                                                         forecast_end_date, forecast_end_date,
                                                                         forecast_config, with_uncertainty=True,
                                                                         include_best_fit=True)

        write_file(to_dict(forecast_config), output_artifacts.M2_forecast_config, "json", "dict", indent=4)

        write_file(df_predictions_forecast_m2, output_artifacts.M2_full_output_forecast_file, "csv", "dataframe")

        # Get percentiles to be plotted
        uncertainty_parameters = forecast_config.model_parameters['uncertainty_parameters']
        confidence_intervals = []
        for c in uncertainty_parameters['confidence_interval_sizes']:
            confidence_intervals.extend([50 - c / 2, 50 + c / 2])
        percentiles = list(set(uncertainty_parameters['percentiles'] + confidence_intervals))
        column_tags = [str(i) for i in percentiles]
        column_tags.extend(['mean', 'best'])
        region_name_str = " ".join(region_name) if isinstance(region_name, list) else region_name

        df_m1_plot, df_m2_plot = dict(), dict()
        df_m1_plot['actual'] = get_observations_subset(df_m1['actual'], plot_start_date_m1, test_end_date)
        df_m1_plot['smoothed'] = get_observations_subset(df_m1['smoothed'], plot_start_date_m1, test_end_date)
        df_m2_plot['actual'] = get_observations_subset(df_m2['actual'], plot_start_date_m2, train2_end_date)
        df_m2_plot['smoothed'] = get_observations_subset(df_m2['smoothed'], plot_start_date_m2, train2_end_date)

        print('Creating plots...')
        # Create M1, M2, M2 forecast plots
        m1_plots(region_name_str, df_m1_plot["actual"], df_m1_plot["smoothed"], df_predictions_train_m1,
                 df_predictions_test_m1, train1_start_date, test_start_date, output_artifacts.dict(),
                 column_tags=column_tags, verbose=verbose)
        m2_plots(region_name_str, df_m2_plot["actual"], df_m2_plot["smoothed"], df_predictions_train_m2,
                 train2_start_date, output_artifacts.dict(),
                 column_tags=column_tags, verbose=verbose)
        m2_forecast_plots(region_name_str, df_m2_plot["actual"], df_m2_plot["smoothed"], df_predictions_forecast_m2,
                          train2_start_date, forecast_start_date, output_artifacts.dict(),
                          column_tags=column_tags, verbose=False)

        # Get trials dataframe
        region_metadata = DataFetcherModule.get_regional_metadata(region_type, region_name, data_source=data_source)
        M2_model = ModelFactory.get_model(model_class, M2_model_params)
        # TODO: LATER - Round 2 refactoring
        trials = M2_model.get_trials_distribution(region_metadata, df_m2["actual"], forecast_run_day,
                                                  forecast_start_date, forecast_end_date)
        # Plot PDF and CDF
        distribution_plots(trials, uncertainty_parameters['variable_of_interest'], output_artifacts.dict())

    @staticmethod
    def generate_planning_outputs_static(region_name, region_type, data_source, input_file_path,
                                         time_interval_config, model_class, uncertainty_parameters, planning,
                                         staffing, model_file, output_artifacts, output_dir):
        """

        Args:
            region_name (list): Region of interest (list of regions possibly singleton) for building the forecasting models
            region_type (str): Type of region [district, state]
            data_source (DataSource): Source of input data [direct_csv, official_data, rootnet_stats_history, tracker_data_all] (default tracker_data_all)
            input_file_path (str): Path to the input csv if data_source is set to direct_csv
            time_interval_config (dict): Dict comprising all the key dates or alternately interval durations for training, evaluation, forecast and plotting. See Session_Definition for more info.
            model_class (ModelClass): Model class of the ensemble model [homogeneous_ensemble, heterogeneous_ensemble]
            uncertainty_parameters (dict): Dict comprising parameters such as confidence_interval, percentiles required for generating a forecast distribution. See Session_Definition for more info.
            planning (dict): Dict comprising parameters such as planning level and scenarios that are required for planning. See Session_Definition for more info.
            staffing (dict): Dict comprising parameters such as staffing ratios, bed type ratios that are required to estimate staffing. See Session_Definition for more info.
            model_file (str): 
            output_artifacts (ModelBuildingSessionOutputArtifacts): List of artifacts generated in the model building session (model jsons, pngs, csvs)
            output_dir (str): Path to the output directory for saving artifacts, trained models, forecasts, plots and reports

        Returns:

        """
        outputs = {}

        # 1. read the ensemble model file
        M2_model_params = read_file(model_file, "json", "dict")
        M2_model = ModelFactory.get_model(model_class, M2_model_params)

        dates = time_interval_config['direct']

        # 2. get the relevant data
        regional_data = DataFetcherModule.get_regional_data(region_type, [region_name], data_source=data_source,
                                                            input_file_path=input_file_path)

        # 3. get the representative model for the chosen planning level
        percentile_params = M2_model.get_params_for_percentiles(
            variable_of_interest=uncertainty_parameters['variable_of_interest'],
            date_of_interest=uncertainty_parameters['date_of_interest'],
            tolerance=uncertainty_parameters['tolerance'], percentiles=[planning['ref_level']],
            region_metadata=regional_data['metadata'], region_observations=regional_data['actual'],
            run_day=dates['forecast_run_day'], start_date=dates['forecast_start_date'],
            end_date=dates['forecast_end_date'])

        M2_planning_model_params = percentile_params[planning['ref_level']]
        write_file(M2_planning_model_params, output_artifacts.M2_planning_model_params, "json", "dict")

        # 4. generate the required forecast config for the child models
        child_model_class = M2_model.child_model_class
        m2_child_forecast_config = ModelBuildingSession._generate_model_operations_config(
            region_name, region_type, data_source, input_file_path, time_interval_config, child_model_class,
            None, None, None, None, None, None, output_artifacts.M2_planning_model_params, "forecast", output_dir)

        # 5. compute the parameters (just r0) for the different scenarios including the planning one
        rt_multiplier_list = [1]
        rt_multiplier_list.extend(planning['rt_multiplier_list'])
        rt_list = [r * M2_planning_model_params['r0'] for r in rt_multiplier_list]
        metrics = {"M2_scenarios_r0": rt_list}

        # 6. loop through the different cases
        # TODO: LATER V3 generalize this piece for handling more complex scenarios - later V3
        forecasting_output = pd.DataFrame()
        for i in range(len(rt_list)):

            #  choose the names for each case - for artifact files
            if i == 0:
                case_name = "planning"
            else:
                case_name = "scenario_" + str(i)

            #  set the model params correctly
            m2_child_forecast_config.model_parameters["r0"] = rt_list[i]

            #  compute the case count forecast
            df_m2 = DataFetcherModule.get_actual_smooth_for_region(region_type, region_name, data_source,
                                                                   input_file_path)

            forecast_m2 = ForecastingModule.flexible_forecast(df_m2["actual"], M2_planning_model_params,
                                                              dates["forecast_run_day"], dates["forecast_start_date"],
                                                              dates["forecast_end_date"], dates["forecast_end_date"],
                                                              m2_child_forecast_config)

            #  compute and save the case count forecast
            forecasting_output = ForecastingModule.from_config(m2_child_forecast_config)
            forecasting_output.to_csv(output_artifacts.__getattribute__(f'staffing_{case_name}'), index=False)

            # compute and save the staffing matrices
            active_count = float(
                forecasting_output[forecasting_output.index == uncertainty_parameters['date_of_interest']][
                    'active_mean'])
            staffing_df = domain_info.compute_staffing_matrix(active_count, staffing['bed_type_ratio'],
                                                              staffing['staffing_ratios_file'],
                                                              staffing['bed_multiplier_count'])
            staff_matrix_file_key = "staffing_" + case_name
            staffing_df.to_csv(output_artifacts.__getattribute__(staff_matrix_file_key), index=False)

            vertical_lines = [
                {'date': dates["train2_start_date"], 'color': 'brown', 'label': 'Train starts'},
                {'date': dates["forecast_start_date"], 'color': 'black', 'label': 'Forecast starts'}
            ]
            df_m2_plot = dict()
            df_m2_plot['actual'] = get_observations_subset(df_m2['actual'], dates["plot_start_date_m2"],
                                                           dates["train2_end_date"])
            df_m2_plot['smoothed'] = get_observations_subset(df_m2['smoothed'], dates["plot_start_date_m2"],
                                                             dates["train2_end_date"])

            multivariate_case_count_plot(df_m2_plot["actual"], df_m2_plot["smoothed"], df_predictions_train=None,
                                         df_predictions_test=forecast_m2, vertical_lines=vertical_lines,
                                         title=f'{region_name}: Scenario {case_name} - M2 forecast',
                                         path=output_artifacts.__getattribute__(f'plot_M2_{case_name}_CARD'))

            forecasting_output = pd.concat([forecasting_output, forecast_m2], axis=1)

        write_file(forecasting_output, output_artifacts.M2_planning_output_forecast_file, "csv", "dataframe")

        outputs['metrics'] = metrics
        return outputs

    @staticmethod
    def list_outputs_static(output_artifacts, display_list=['model_building_report', 'planning_report']):
        """

        Args:
            output_artifacts (ModelBuildingSessionOutputArtifacts): List of artifacts generated in the model building session (model jsons, pngs, csvs)
            display_list (list): 

        Returns:

        """
        outputs = {}
        print('Listing all the output artifacts from the session:\n')
        for key, value in output_artifacts.__fields__.items():
            print(key + ': ' + value.name + '\n')

        for a in display_list:
            render_artifact(output_artifacts.__getattribute__(a), ".md")
        return outputs

    @staticmethod
    def log_session_static(params, metrics, output_artifacts, experiment_name, session_name):
        """

        Args:
            params (ModelBuildingSessionParams): Dict comprising input parameters specified by the user for the session. See Session_Definition for more info.
            metrics (ModelBuildingSessionMetrics): Dict comprising output metrics such as losses associated with the model building. See Session_Definition for more info.
            output_artifacts (ModelBuildingSessionOutputArtifacts): List of artifacts generated in the model building session (model jsons, pngs, csvs)
            experiment_name (str): MLFlow experiment name
            session_name (str): Session name variable

        Returns:

        """
        input_artifacts = {input_artifact.split(".")[-1]: get_dict_field(params.dict(), input_artifact.split(".")) for
                           input_artifact in params.input_artifacts}
        artifacts_to_log = output_artifacts.dict()
        artifacts_to_log.update(input_artifacts)
        outputs = None
        try:
            outputs = mlflow_logger.log_to_mlflow(params, metrics, artifacts_to_log,
                                                  experiment_name=experiment_name, run_name=session_name)
        except:
            pass
        return outputs
