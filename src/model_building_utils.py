import json
import os

import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from copy import deepcopy
from datetime import datetime, timedelta

from configs.base_config import TrainingModuleConfig
from configs.base_config import ModelEvaluatorConfig

from modules.data_fetcher_module import DataFetcherModule
from modules.forecasting_module import ForecastingModule
from configs.base_config import ForecastingModuleConfig
from modules.model_evaluator import ModelEvaluator
from modules.training_module import TrainingModule
from publishers.mlflow_logging import get_previous_runs

import matplotlib.dates as mdates

from nb_utils import plot_data
from nb_utils import train_eval_plot_ensemble

class ModelBuildingSession:
    def __init__(self, sess_name):
        self.sess_name = sess_name
        self.params = dict()
        self.load_default_configs()
        self.init_mandatory_params()
        self.init_config_params()
        self.init_mlflow_athena_config()
        
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

        self.params['train_loss_function_config.C_weight'] = 0.25
        self.params['train_loss_function_config.A_weight'] = 0.25
        self.params['train_loss_function_config.R_weight'] = 0.25
        self.params['train_loss_function_config.D_weight'] = 0.25

        # Will set the train, forecast intervals
        # TODO: Need to set the train_eval_plot_ensemble() function to
        # accept the right intervals
        # Currently the function takes in a deterministic interval on the train duration
        self.params['time_interval_config.train_start_date'] = datetime.now().date()
        self.params['time_interval_config.train_end_date'] = datetime.now().date()
        self.params['time_interval_config.backtesting_split_date'] = datetime.now().date()
        self.params['time_interval_config.forecast_start_date'] = datetime.now().date() 
        self.params['time_interval_config.forecast_end_date'] = datetime.now().date()
        self.params['time_interval_config.forecast_planning_date'] = datetime.now().date()
    
    def print_parameters(self):
        for param in self.params:
            print(f'{param} : {self.params[param]}')
    
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
            self.params['case_cnt_plot_file_name'] = plot_name

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
        dir_prefix = '../notebooks'

        plot_data(region, region_type, output_dir_name, dir_prefix, data_source=data_source, data_path=data_path, 
                  plot_config='../notebooks/plot_config.json', plot_name=plot_fname, csv_name=csv_fname)
        

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

    
    def build_models_and_generate_forecast(self):
        mandatory_params = self.mandatory_params['build_models_and_generate_forecast']
        self.validate_params(mandatory_params)
        # Use the region and all arguments to train and 
        # generate forecasts
        
        # train_config[''] = 'blah'
        # train_eval_forecast() # equivalent from nb_utils

        self.setup_defaults()

        current_day = datetime.now().date() 
        forecast_length = 30
        
        #TODO check the date_of_interest parameter
        self.forecast_config['model_parameters']['uncertainty_parameters']['date_of_interest'] = (current_day + timedelta(forecast_length/2)).strftime("%-m/%-d/%y")


        name_prefix = self.params['region_name']
        
        train_eval_plot_ensemble([self.params['region_name']], self.params['region_type'],
                         current_day, forecast_length,
                         self.train_config, self.test_config, self.forecast_config,
                         train_period = 14, test_period = 7,
                         max_evals = 1000, data_source = self.params['data_source'],
                         input_filepath = self.params['data_filepath'],
                         mlflow_log = False, mlflow_run_name = "Testing combined data")
        #return params, metrics
    
    def generate_planning_outputs(self):
        mandatory_params = self.mandatory_params['generate_planning_outputs']
        self.validate_params(mandatory_params)
        # Use scenario configs to generate planning outputs
        pass
    
        #TODO: Check if we can get model params corresponding to a percentile level?
        #Given the params correponding to the percentile level, enable a multiplier on
        #r0 to generate a scenario forecast?
    
    def list_outputs(self):
        mandatory_params = self.mandatory_params['list_outputs']
        self.validate_params(mandatory_params)
        csv_fname = self.params['case_cnt_csv_file_name']
        plot_fname = self.params['case_cnt_plot_file_name']
        plt.show(plot_fname)
        return pd.read_csv(csv_fname)
    
    def log_session(self):
        mandatory_params = self.mandatory_params['log_session']
        self.validate_params(mandatory_params)
        pass