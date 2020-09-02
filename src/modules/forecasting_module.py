import os

import pandas as pd
from configs.base_config import ForecastingModuleConfig
from entities.model_class import ModelClass
from model_wrappers.model_factory import ModelFactory
from modules.data_fetcher_module import DataFetcherModule
from utils.data_util import convert_to_old_required_format, convert_to_required_format, \
    add_init_observations_to_predictions, get_date
from utils.io import read_config_file


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
        predictions_df = convert_to_old_required_format(run_day, predictions_df, region_type, region_name,
                                                        self._model_parameters['MAPE'], self._model.__name__)
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
                                                            config.forecast_run_day, config.forecast_start_date,
                                                            config.forecast_end_date, config.input_filepath)
        if config.output_dir is not None and config.output_file_prefix is not None:
            predictions.to_csv(os.path.join(config.output_dir, f'{config.output_file_prefix}.csv'), index=False)
        return predictions

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
