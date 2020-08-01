import os

import pandas as pd
from configs.base_config import ForecastingModuleConfig
from entities.model_class import ModelClass
from model_wrappers.model_factory import ModelFactory
from modules.data_fetcher_module import DataFetcherModule
from utils.config_util import read_config_file
from utils.data_transformer_helper import convert_to_old_required_format, convert_to_required_format


class ForecastingModule(object):

    def __init__(self, model_class: ModelClass, model_parameters: dict):
        self._model_parameters = model_parameters
        self._model = ModelFactory.get_model(model_class, model_parameters)

    def predict(self, region_type: str, region_name: str, region_metadata: dict, region_observations: pd.DataFrame,
                run_day: str, forecast_start_date: str, forecast_end_date: str):
        predictions_df = self._model.predict(region_metadata, region_observations, run_day, forecast_start_date,
                                             forecast_end_date)
        predictions_df = convert_to_required_format(predictions_df, region_type, region_name)
        return predictions_df

    def predict_old_format(self, region_type: str, region_name: str, region_metadata: dict,
                           region_observations: pd.DataFrame,
                           run_day: str, forecast_start_date: str,
                           forecast_end_date: str):
        predictions_df = self._model.predict(region_metadata, region_observations, run_day, forecast_start_date,
                                             forecast_end_date)
        predictions_df = convert_to_old_required_format(run_day, predictions_df, region_type, region_name)
        return predictions_df.to_json()

    def predict_for_region(self, data_source, region_type, region_name, run_day, forecast_start_date,
                           forecast_end_date, input_filepath):
        observations = DataFetcherModule.get_observations_for_region(region_type, region_name, data_source=data_source,
                                                                     filepath=input_filepath)
        region_metadata = DataFetcherModule.get_regional_metadata(region_type, region_name, data_source=data_source)
        return self.predict(region_type, region_name, region_metadata, observations, run_day,
                            forecast_start_date,
                            forecast_end_date)

    @staticmethod
    def from_config_file(config_file_path):
        config = read_config_file(config_file_path)
        forecasting_module_config = ForecastingModuleConfig.parse_obj(config)
        return ForecastingModule.from_config(forecasting_module_config)

    @staticmethod
    def from_config(config: ForecastingModuleConfig):
        forecasting_module = ForecastingModule(config.model_class, config.model_parameters)
        predictions = forecasting_module.predict_for_region(config.data_source, config.region_type, config.region_name,
                                                            config.run_day, config.forecast_start_date,
                                                            config.forecast_end_date, config.input_filepath)
        if config.output_dir is not None and config.output_file_prefix is not None:
            predictions.to_csv(os.path.join(config.output_dir, f'{config.output_file_prefix}.csv'), index=False)
        return predictions
