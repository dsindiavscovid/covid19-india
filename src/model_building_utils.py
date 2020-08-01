import json
import os
from copy import deepcopy
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from IPython.display import Markdown, display
from configs.base_config import ForecastingModuleConfig
from model_wrappers.model_factory import ModelFactory
from modules.data_fetcher_module import DataFetcherModule
from modules.forecasting_module import ForecastingModule
from nb_utils import train_eval_plot_ensemble
from publishers.mlflow_logging import get_previous_runs, log_to_mlflow
from publishers.report_generation import create_report
from utils.data_transformer_helper import flatten_train_loss_config, flatten_eval_loss_config, flatten, \
    get_observations_subset, add_init_observations_to_predictions
from utils.plotting import m2_forecast_plots, plot_data
from utils.staffing import get_clean_staffing_ratio, compute_staffing_matrix


def create_session(session_name, user_name='guest'):
    if session_name is None:
        current_date = datetime.now().date().strftime('%-m/%-d/%y')
        session_name = user_name + current_date
    # Directory to store the outptus from the current session
    dir_name = session_name + '_outputs'
    # Check if the directory exists
    # if yes, raise a warning; else create a new directory
    path_prefix = '../outputs'
    output_dir = os.path.join(path_prefix, dir_name)
    if os.path.isdir(output_dir):
        print(f"{output_dir} already exists. Your logs might get overwritten.")
        print("Please change the session name to retain previous logs")
        pass
    else:
        os.mkdir(output_dir)
    current_session = ModelBuildingSession(session_name)
    current_session.set_param('output_dir', dir_name)
    current_session.set_param('experiment_name', 'SEIHRD_ENSEMBLE_V0')
    current_session.set_param('run_name', session_name)
    current_session.set_param('model_class', 'homogeneous_ensemble')
    current_session.print_parameters()
    #TOFIX
    current_session.forecast_config['model_parameters']['uncertainty_parameters']['date_of_interest'] = '7/5/20'
    return current_session


class ModelBuildingSession:
    """
        Session object to encapsulate the model building
        and training process.


    """
    def __init__(self, sess_name):
        self.sess_name = sess_name
        self.params = dict()
        self.model_params = None  # Placeholder for trained model parameters
        self.load_default_configs()
        self.init_mandatory_params()
        self.init_config_params()
        self.init_mlflow_athena_config()
        self.params_to_log = dict()
        self.metrics_to_log = dict()
        self.artifacts_dict = dict()

    def load_default_configs(self):
        """
        Read from config files?
        """
        with open('../config/sample_homogeneous_train_config.json') as f_train, \
            open('../config/sample_homogeneous_test_config.json') as f_test, \
            open('../config/sample_homogeneous_forecast_config.json') as f_forecast:
            self.train_config = json.load(f_train)
            self.test_config = json.load(f_test)
            self.forecast_config = json.load(f_forecast)

        self.time_interval_config = None
        self.param_search_config = None
        self.train_loss_config = None
        self.eval_loss_config = None
    
    def init_mlflow_athena_config(self):
        with open('/Users/shreyas.shetty/mlflow_config.json') as f_mlflow:
            mlflow_config = json.load(f_mlflow)
        mlflow.set_tracking_uri(mlflow_config['tracking_url'])
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_config['username']
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_config['password']

        # Source the pyathena configuration
        os.system('source ../../pyathena/pyathena.rc')


    def init_mandatory_params(self):
        self.mandatory_params = dict()
        self.mandatory_params['view_forecasts'] = ['experiment_name', 'region_type', 'region_name']
        self.mandatory_params['examine_data'] = ['region_type', 'region_name']
        self.mandatory_params['build_models_and_generate_forecast'] = ['model_class']
        self.mandatory_params['generate_planning_outputs'] = []
        self.mandatory_params['list_outputs'] = []
        self.mandatory_params['log_session'] = []

    def init_config_params(self):
        self.params['experiment_name'] = 'SEIHRD_ENSEMBLE_V0'
        
        #TODO: Load Athena config params

        self.params['data_filepath'] = None
        self.params['planning_variable'] = 'confirmed'

        self.params['train_loss_function_config.C_weight'] = 0.25
        self.params['train_loss_function_config.A_weight'] = 0.25
        self.params['train_loss_function_config.R_weight'] = 0.25
        self.params['train_loss_function_config.D_weight'] = 0.25

        # Will set the train, forecast intervals
        # TODO: Need to set the train_eval_plot_ensemble() function to
        # accept the right intervals
        # Currently the function takes in a deterministic interval on the train duration
        #self.params['time_interval_config.train_start_date'] = datetime.now().date()
        #self.params['time_interval_config.train_end_date'] = datetime.now().date()
        #self.params['time_interval_config.backtesting_split_date'] = datetime.now().date()
        #self.params['time_interval_config.forecast_start_date'] = datetime.now().date() 
        #self.params['time_interval_config.forecast_end_date'] = datetime.now().date()
        #self.params['time_interval_config.forecast_planning_date'] = datetime.now().date()

        self.params['bed_type_ratio'] = {'CCC2':0.62, 'DCHC':0.17,'DCH':0.16,'ICU':0.05}
        self.params['bed_multiplier_count'] = 100
        self.params['planning_level'] = 80
        self.params['rt_multiplier_list'] = [0.9, 1.1, 1.2]
    
    def print_parameters(self):
        for param in self.params:
            print('{} : {}'.format(param, self.params[param]))
    
    def validate_params(self, param_list):
        """
        Validation should include the particular function
        we want to validate.
        Need to work out the details.
        
        Pydantic?

        For now; implemented as a check on the presence of all the 
        params from the pre-defined list of parameters corresponding 
        to each function.
        """
        for param in param_list:
            if param not in self.params:
                raise Exception('Please check the list of parameters')
    
    def set_param(self, param, param_value):
        """
        Q: How to we allow for setting config params?
        For example: train_loss_function_config.C_weight?
        """
        # Some more type checking of sorts
        # I should not be able to change already set parameters

        # TODO: Check with parameter override options
        #if param in self.params:
        #    return # Will address the override param functionality later
        #else:
        #    self.params[param] = param_value

        self.params[param] = param_value
    
    def view_forecasts(self):
        mandatory_params = self.mandatory_params['view_forecasts']
        self.validate_params(mandatory_params)
        # Use intervals, region_type and region_name
        # as arguments to the forecast function
        
        # Where will we get the stored models from?
        pass
        #TODO: Do we have separate experiments for individual cities?
        #for now, assume a common experiment name; depending on the model_class
        experiment_name = self.params['experiment_name']
        region_name = self.params['region_name']

        if 'interval_to_consider' in self.params:
            interval = self.params['interval_to_consider']
        else:
            interval = 0

        links_df = get_previous_runs(experiment_name, region_name, interval)

        return links_df
    
    def examine_data(self):
        mandatory_params = self.mandatory_params['examine_data']
        self.validate_params(mandatory_params)
        # Get data from data_fetcher module and 
        # generate outputs csv's in the appropriate paths
        
        # Where do we get the input csv if the user wants to upload one?
        if 'case_cnt_plot_file_name' in self.params:
            plot_fname = self.params['case_cnt_plot_file_name']
        else:
            plot_fname = 'case_cnt_plot.png'
            self.params['case_cnt_plot_file_name'] = plot_fname

        if 'case_cnt_csv_file_name' in self.params:
            csv_fname = self.params['case_cnt_csv_file_name']
        else:
            csv_fname = 'case_cnt.csv'
            self.params['case_cnt_csv_file_name'] = csv_fname

        region_type = self.params['region_type']
        region = self.params['region_name']
        data_source = self.params['data_source']
        data_path = self.params['data_filepath']
        output_dir_name = self.params['output_dir']
        dir_prefix = '../outputs'

        plot_data(region, region_type, output_dir_name, dir_prefix=dir_prefix, data_source=data_source,
                  data_path=data_path, plot_name=plot_fname, csv_name=csv_fname)
        

    def setup_defaults(self):
        model_class = self.params['model_class'] #'homogeneous_ensemble'
        self.train_config['model_class'] = model_class
        self.test_config['model_class'] = model_class
        self.forecast_config['model_class'] = model_class

        c_weight = self.params['train_loss_function_config.C_weight']
        a_weight = self.params['train_loss_function_config.A_weight']
        r_weight = self.params['train_loss_function_config.R_weight']
        d_weight = self.params['train_loss_function_config.D_weight']

        # The default config file uses a list of dictionaries for the 
        # weights on the individual buckets in the order of 'c', 'a', 'r', 'd'
        self.train_config['training_loss_function']['variable_weights'][0]['weight'] = c_weight
        self.train_config['training_loss_function']['variable_weights'][1]['weight'] = a_weight
        self.train_config['training_loss_function']['variable_weights'][2]['weight'] = r_weight
        self.train_config['training_loss_function']['variable_weights'][3]['weight'] = d_weight

        #self.train_config['train_start_date'] = self.params['time_interval_config.train_start_date']
        #self.train_config['train_end_date'] = self.params['time_interval_config.train_end_date']
        #backtesting_split_date = self.params['time_interval_config.backtesting_split_date']

        forecast_start_date = self.params['time_interval_config.forecast_start_date']
        forecast_end_date = self.params['time_interval_config.forecast_end_date']
        self.forecast_config['forecast_start_date'] = forecast_start_date
        self.forecast_config['forecast_end_date'] = forecast_end_date

        forecast_planning_date = self.params['time_interval_config.forecast_planning_date']
        self.forecast_config['model_parameters']['uncertainty_parameters']['date_of_interest'] = forecast_planning_date

    
    def render_report(self, path):
        with open(path) as fh:
            content = fh.read()

        display(Markdown(content))


    def build_models_and_generate_forecast(self):
        mandatory_params = self.mandatory_params['build_models_and_generate_forecast']
        self.validate_params(mandatory_params)


        self.setup_defaults()
        self.param_search_config = self.train_config['search_space']
        self.train_loss_config = self.train_config['training_loss_function']
        self.eval_loss_config = self.train_config['loss_functions']
        model_class = self.params['model_class']
        
        # Region params
        region = self.params['region_name']
        region_type = self.params['region_type']

        # Train/test intervals
        train1_start_date = self.params['time_interval_config.train_start_date']
        train1_run_day = (datetime.strptime(train1_start_date, "%m/%d/%y") - timedelta(1)).strftime("%-m/%-d/%y")
        backtesting_split_date = self.params['time_interval_config.backtesting_split_date']
        train1_end_date = (datetime.strptime(backtesting_split_date, "%m/%d/%y") - timedelta(1)).strftime("%-m/%-d/%y")

        test_run_day = (datetime.strptime(backtesting_split_date, "%m/%d/%y") - timedelta(1)).strftime("%-m/%-d/%y")
        test_start_date = backtesting_split_date
        # Use the last date of the train interval as the test interval end date
        test_end_date = self.params['time_interval_config.train_end_date']

        train2_run_day = (datetime.strptime(backtesting_split_date, "%m/%d/%y") - timedelta(1)).strftime("%-m/%-d/%y") 
        train2_start_date = self.params['time_interval_config.backtesting_split_date']
        train2_end_date = self.params['time_interval_config.train_end_date']

        forecast_start_date = self.params['time_interval_config.forecast_start_date']
        forecast_run_day = (datetime.strptime(forecast_start_date, "%m/%d/%y") - timedelta(1)).strftime("%-m/%-d/%y")
        forecast_end_date = self.params['time_interval_config.forecast_end_date']

        name_prefix = self.params['region_name']
        output_dir = os.path.join('../outputs', self.params['output_dir'])

        params, metrics, artifacts_dict, train1_params, train2_params = train_eval_plot_ensemble(region, region_type,
            train1_run_day, train1_start_date, train1_end_date,
            test_run_day, test_start_date, test_end_date,
            train2_run_day, train2_start_date, train2_end_date,
            forecast_run_day, forecast_start_date, forecast_end_date,
            self.train_config, self.test_config, 
            self.forecast_config, 
            data_source=self.params['data_source'], 
            input_filepath=self.params['data_filepath'], 
            output_dir=output_dir, 
            mlflow_log=False, 
            mlflow_run_name=self.params['run_name'])


        m1_model_params = train1_params['model_parameters']
        m1_model = ModelFactory.get_model(model_class, m1_model_params)
        m1_metrics_param_ranges = m1_model.get_statistics_of_params()
        m2_model_params = train2_params['model_parameters']
        m2_model = ModelFactory.get_model(model_class, m2_model_params)
        m2_metrics_param_ranges = m2_model.get_statistics_of_params()

        metrics['M1_param_ranges'] = m1_metrics_param_ranges
        metrics['M2_param_ranges'] = m2_metrics_param_ranges

        # Persist model parameters for generating planning reports
        self.m2_model_params = m2_model_params

        self.time_interval_config = params['time_interval_config']

        model_building_report = os.path.join(output_dir, 'model_building_report.md')
        create_report(params, metrics, artifacts_dict,
                      template_path='publishers/model_building_template_v1.mustache',
                      report_path=model_building_report)

        self.params_to_log.update(params)
        self.metrics_to_log.update(metrics)
        self.artifacts_dict.update(artifacts_dict)

        return params, metrics, train1_params, train2_params
    
    def generate_planning_plots(self, model_params, planning_variable, 
                                planning_level, planning_date, 
                                region_type, region_name, 
                                data_source, input_filepath, 
                                train2_start_date, train2_end_date, 
                                forecast_run_day, forecast_start_date, 
                                forecast_end_date):


        params = dict()
        metrics = dict()
        artifacts_dict = dict()
        r0 = model_params['r0']
        params['uncertainty_config'] = self.forecast_config['model_parameters']['uncertainty_parameters']
        params['region_type'] = region_type
        params['region_name'] = region_name
        params['forecast_planning_variable'] = planning_variable
        params['forecast_planning_date'] = planning_date
        params['scenario_config_R0_multipliers'] = self.params['rt_multiplier_list']
        metrics['M2_planning_level_model'] = model_params
        r0_list = [r*r0 for r in self.params['rt_multiplier_list']]
        metrics['M2_whatif_scenarios_R0'] = r0_list
        staffing_ratios_file_path = self.params['staff_ratios_file_path']


        # Get staffing ratios
        staff_ratios = get_clean_staffing_ratio(staffing_ratios_file_path)
        bed_type_ratio = self.params['bed_type_ratio']
        bed_multiplier_count = self.params['bed_multiplier_count']
        metrics['staffing_planning'] = staff_ratios

        with open('../config/sample_forecasting_config.json') as f:
            forecast_config = json.load(f)

        # Get actual and smoothed observations
        df_actual = DataFetcherModule.get_observations_for_region(region_type, [region_name], 
                                                                  data_source=data_source, 
                                                                  smooth=False, filepath=input_filepath)
        df_smoothed = DataFetcherModule.get_observations_for_region(region_type, [region_name], 
                                                                    data_source=data_source, smooth=True, 
                                                                    filepath=input_filepath)
        plot_start_date_m2 = (datetime.strptime(train2_start_date, "%m/%d/%y") - timedelta(days=7)).strftime("%-m/%-d/%y")
        df_actual_m2 = get_observations_subset(df_actual, plot_start_date_m2, train2_end_date)
        df_smoothed_m2 = get_observations_subset(df_smoothed, plot_start_date_m2, train2_end_date)


        rt_multiplier_list = [1] + self.params['rt_multiplier_list']    
        for idx, r0_mult in enumerate(rt_multiplier_list):
            percentile_config = deepcopy(forecast_config)
            forecast_module = ForecastingModuleConfig.parse_obj(percentile_config)

            # Fetch the model parameters and update it using the 
            # multiplication factor
            model_params['r0'] = model_params['r0'] * r0_mult
            forecast_module.model_parameters = model_params

            forecast_module.data_source = self.params['data_source']
            forecast_module.input_filepath = self.params['data_filepath']
            forecast_module.model_class = 'SEIHRD_gen'
            forecast_module.run_day = forecast_run_day
            forecast_module.forecast_start_date = forecast_start_date
            forecast_module.forecast_end_date = forecast_end_date
    
            forecasting_output = ForecastingModule.from_config(forecast_module)
            output_dir = self.params['output_dir']
            artifact_name = 'forecast_' + str(planning_level) + '_' + str(r0_mult) + '.csv'
            planning_outputs = os.path.join('../outputs', output_dir, 'planning_outputs')
            artifact_path = os.path.join(planning_outputs, artifact_name)
            forecasting_output.to_csv(artifact_path, index=False)

            forecasting_output = forecasting_output.reset_index()

            if idx >= 1:
                active_count = float(forecasting_output[forecasting_output['date'] == planning_date]['active_mean'])
                staffing_df = compute_staffing_matrix(active_count,bed_type_ratio,
                                                      staffing_ratios_file_path,
                                                      bed_multiplier_count)
                metrics[f'staffing_scenario_{idx}'] = staffing_df


            df_predictions_forecast_m2 = add_init_observations_to_predictions(df_actual, forecasting_output,
                                                                  forecast_run_day)
            df_predictions_forecast_m2['date'] = pd.to_datetime(df_predictions_forecast_m2['date'], format='%m/%d/%y')

            if not os.path.exists(planning_outputs):
                os.mkdir(planning_outputs)
            m2_forecast_plots(region_name, df_actual_m2, df_smoothed_m2, df_predictions_forecast_m2,
                              train2_start_date, forecast_start_date, column_tags=['mean'], output_dir=planning_outputs,
                              debug=False, scenario=str(planning_level) + '_' + str(r0_mult))

        
        artifacts_dict = {
            'plot_M2_planning_CARD': os.path.join(planning_outputs,f'{planning_level}_1_m2_forecast.png'),
            'plot_M2_scenario_1_CARD': os.path.join(planning_outputs,f'{planning_level}_0.9_m2_forecast.png'),
            'plot_M2_scenario_2_CARD': os.path.join(planning_outputs,f'{planning_level}_1.1_m2_forecast.png'),
            'plot_M2_scenario_3_CARD': os.path.join(planning_outputs,f'{planning_level}_1.2_m2_forecast.png'),
            'planning_output_forecast_file' : os.path.join(planning_outputs, f'forecast_{planning_level}_1.csv')
        }

        # self.artifacts_dict.update(artifacts_dict)

        planning_report = os.path.join('../outputs', output_dir, 'planning_report.md')
        create_report(params, metrics, artifacts_dict, 
                      template_path='publishers/planning_template_v1.mustache', 
                      report_path=planning_report)
        planning_artifacts_list = list(artifacts_dict.values())
        planning_artifacts_list.append(planning_report)
        return planning_artifacts_list

    
    def generate_planning_outputs(self):
        mandatory_params = self.mandatory_params['generate_planning_outputs']
        self.validate_params(mandatory_params)
        # Use scenario configs to generate planning outputs
        pass
    
        #TODO: Check if we can get model params corresponding to a percentile level?
        #Given the params correponding to the percentile level, enable a multiplier on
        #r0 to generate a scenario forecast?

        model_class = self.params['model_class']
        model_params = self.m2_model_params
        region_type = self.params['region_type']
        region_name = self.params['region_name']
        data_source = self.params['data_source']
        input_filepath = self.params['data_filepath']
        train2_start_date = self.time_interval_config['train2_start_date']
        train2_end_date = self.time_interval_config['train2_end_date']
        forecast_run_day = self.time_interval_config['forecast_run_day']
        forecast_start_date = self.time_interval_config['forecast_start_date']
        forecast_end_date = self.time_interval_config['forecast_end_date']
        planning_level = self.params['planning_level']
        planning_date = self.forecast_config['model_parameters']['uncertainty_parameters']['date_of_interest']

        model = ModelFactory.get_model(model_class, model_params)

        observations = DataFetcherModule.get_observations_for_region(region_type, [region_name], data_source=data_source, filepath=input_filepath)
        region_metadata = DataFetcherModule.get_regional_metadata(region_type, [region_name], data_source=data_source)
        planning_variable = 'confirmed' #self.params['planning_variable']
        percentile_params = model.get_params_for_percentiles(column_of_interest = planning_variable, 
                                 date_of_interest = planning_date, 
                                 tolerance = 1, 
                                 percentiles = [planning_level], 
                                 region_metadata = region_metadata, 
                                 region_observations= observations,
                                 run_day = forecast_run_day, 
                                 start_date = forecast_start_date, 
                                 end_date = forecast_end_date)

        planning_model_params = percentile_params[planning_level]

        planning_artifact_list = self.generate_planning_plots(planning_model_params, planning_variable, 
                                                              planning_level, planning_date,
                                                              region_type, region_name, 
                                                              data_source, input_filepath, 
                                                              train2_start_date, train2_end_date, 
                                                              forecast_run_day, forecast_start_date, forecast_end_date)

        return planning_artifact_list
    
    def list_outputs(self):
        mandatory_params = self.mandatory_params['list_outputs']
        self.validate_params(mandatory_params)
        csv_fname = self.params['case_cnt_csv_file_name']
        plot_fname = self.params['case_cnt_plot_file_name']
        plt.show(plot_fname)
        return pd.read_csv(csv_fname)
    
    def log_session(self):
        # TODO: Log comments and questions
        mandatory_params = self.mandatory_params['log_session']
        self.validate_params(mandatory_params)

        try:
            self.params_to_log['train_loss_function_config'] = \
                flatten_train_loss_config(self.params_to_log['train_loss_function_config'])
        except KeyError:
            pass
        try:
            self.params_to_log['eval_loss_function_config'] = \
                flatten_eval_loss_config(self.params_to_log['eval_loss_function_config'])
        except KeyError:
            pass
        self.params_to_log = flatten(self.params_to_log)
        self.metrics_to_log = flatten(self.metrics_to_log)

        remove_keys = []
        for key, value in self.metrics_to_log.items():
            if isinstance(value, pd.DataFrame):  # TODO: Check for numeric type
                remove_keys.append(key)

        [self.metrics_to_log.pop(key) for key in remove_keys]

        log_to_mlflow(self.params_to_log, self.metrics_to_log, self.artifacts_dict,
                      experiment_name=self.params['experiment_name'], run_name=self.params['run_name'])

