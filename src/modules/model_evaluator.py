import json

from configs.base_config import ModelEvaluatorConfig
from entities.model_class import ModelClass
from model_wrappers.model_factory import ModelFactory
from modules.data_fetcher_module import DataFetcherModule
from utils.config_util import read_config_file
from utils.loss_util import evaluate_for_forecast


class ModelEvaluator(object):

    def __init__(self, model_class: ModelClass, model_parameters: dict):
        self._model = ModelFactory.get_model(model_class, model_parameters)

    def evaluate(self, region_metadata, observations, run_day, test_start_date, test_end_date, loss_functions):
        predictions = self._model.predict(region_metadata, observations, run_day, test_start_date, test_end_date)
        return evaluate_for_forecast(observations, predictions, loss_functions)

    def evaluate_for_region(self, data_source, region_type, region_name, run_day, test_start_date, test_end_date, loss_functions):
        observations = DataFetcherModule.get_observations_for_region(region_type, region_name, data_source)
        region_metadata = DataFetcherModule.get_regional_metadata(region_type, region_name, data_source)
        return self.evaluate(region_metadata, observations, run_day, test_start_date, test_end_date, loss_functions)

    @staticmethod
    def from_config(config: ModelEvaluatorConfig):
        model_evaluator = ModelEvaluator(config.model_class, config.model_parameters)
        metric_results = model_evaluator.evaluate_for_region(config.data_source, config.region_type, config.region_name,
                                                             config.run_day, config.test_start_date,
                                                             config.test_end_date, config.loss_functions)
        if config.output_filepath is not None:
            with open(config.output_filepath, 'w') as outfile:
                json.dump(metric_results, outfile, indent=4)
        return metric_results

    @staticmethod
    def from_config_file(config_file_path: str):
        config = read_config_file(config_file_path)
        model_evaluator_config = ModelEvaluatorConfig.parse_obj(config)
        return ModelEvaluator.from_config(model_evaluator_config)
