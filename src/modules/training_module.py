from datetime import datetime, timedelta

from configs.base_config import TrainingModuleConfig
from model_wrappers.model_factory import ModelFactory
from modules.data_fetcher_module import DataFetcherModule
from modules.model_evaluator import ModelEvaluator
from utils.io import read_config_file


class TrainingModule(object):

    def __init__(self, model_class, model_parameters):
        self._model = ModelFactory.get_model(model_class, model_parameters)
        self._model_class = model_class
        self._model_parameters = model_parameters

    def train(self, region_metadata, region_observations, train_start_date, train_end_date, search_space,
              search_parameters, train_loss_function):
        return self._model.train(region_metadata, region_observations, train_start_date, train_end_date,
                                 search_space, search_parameters, train_loss_function)

    def train_for_region(self, data_source, region_type, region_name, train_start_date, train_end_date,
                         search_space, search_parameters, train_loss_function, input_filepath):
        observations = DataFetcherModule.get_observations_for_region(region_type, region_name,
                                                                     data_source=data_source, filepath=input_filepath)
        region_metadata = DataFetcherModule.get_regional_metadata(region_type, region_name, data_source=data_source)
        return self.train(region_metadata, observations, train_start_date, train_end_date,
                          search_space, search_parameters, train_loss_function)

    @staticmethod
    def from_config(config: TrainingModuleConfig):
        training_module = TrainingModule(config.model_class, config.model_parameters)
        results = training_module.train_for_region(config.data_source, config.region_type, config.region_name,
                                                   config.train_start_date,
                                                   config.train_end_date,
                                                   config.search_space,
                                                   config.search_parameters, config.train_loss_function,
                                                   config.input_filepath)

        if not ("modes" in config.model_parameters.keys() and config.model_parameters['modes'][
            'training_mode'] == 'constituent_models'):
            config.model_parameters.update(
                results["model_parameters"])  # updating model parameters with best params found above
            model_evaluator = ModelEvaluator(config.model_class, config.model_parameters)
            run_day = (datetime.strptime(config.train_start_date, "%m/%d/%y") - timedelta(days=1)).strftime(
                "%-m/%-d/%y")
            results["train_metric_results"] = model_evaluator.evaluate_for_region(config.data_source,
                                                                                  config.region_type,
                                                                                  config.region_name, run_day,
                                                                                  config.train_start_date,
                                                                                  config.train_end_date,
                                                                                  config.eval_loss_functions,
                                                                                  config.input_filepath)

        results['model'] = training_module._model
        return results

    @staticmethod
    def from_config_file(config_file_path: str):
        config = read_config_file(config_file_path)
        training_module_config = TrainingModuleConfig.parse_obj(config)
        return TrainingModule.from_config(training_module_config)
