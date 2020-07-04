import pandas as pd
import numpy as np

from copy import deepcopy
from datetime import timedelta, datetime
from functools import reduce, partial
from hyperopt import hp
from cachetools import cached, TTLCache

from entities.forecast_variables import ForecastVariable
from entities.loss_function import LossFunction
from model_wrappers.base import ModelWrapperBase
import model_wrappers.model_factory as model_factory_alias
from utils.ensemble_util import get_weighted_predictions, create_trials_dataframe, uncertainty_dict_to_df
from utils.distribution_util import weights_to_pdf, pdf_to_cdf, get_best_index
from utils.hyperparam_util import hyperparam_tuning
from utils.loss_util import evaluate_for_forecast


class HeterogeneousEnsemble(ModelWrapperBase):

    def __init__(self, model_parameters):
        self.model_parameters = model_parameters
        self.models = deepcopy(model_parameters['constituent_models'])
        self.losses = deepcopy(model_parameters['constituent_model_losses'])

        if 'constituent_model_weights' in model_parameters:
            self.weights = deepcopy(model_parameters['constituent_model_weights'])
        else:
            self.weights = None

        for idx in self.models:
            constituent_model = self.models[idx]
            constituent_model_class = constituent_model['model_class']
            constituent_model_parameters = constituent_model['model_parameters']
            self.models[idx] = model_factory_alias.ModelFactory.get_model(
                constituent_model_class, constituent_model_parameters)

    def supported_forecast_variables(self):
        return [ForecastVariable.confirmed, ForecastVariable.recovered, ForecastVariable.active]

    def fit(self):
        raise BaseException("Not supported for ensemble model")

    def is_black_box(self):
        return True

    def train(self, region_metadata: dict, region_observations: pd.DataFrame, train_start_date: str,
              train_end_date: str, search_space: dict, search_parameters: dict, train_loss_function: LossFunction):
        result = {}
        if self.is_black_box():
            objective = partial(self.optimize, region_metadata=region_metadata, region_observations=region_observations,
                                train_start_date=train_start_date, train_end_date=train_end_date,
                                loss_function=train_loss_function)
            for k, v in search_space.items():
                search_space[k] = hp.uniform(k, v[0], v[1])
            result = hyperparam_tuning(objective, search_space,
                                       search_parameters.get("max_evals", 100))

        model_params = self.model_parameters
        model_params.update(result["best_params"])
        model_params["MAPE"] = result["best_loss"]
        result["model_parameters"] = model_params
        return {"model_parameters": model_params}

    def optimize(self, search_space, region_metadata, region_observations, train_start_date, train_end_date,
                 loss_function):
        run_day = (datetime.strptime(train_start_date, "%m/%d/%y") - timedelta(days=1)).strftime("%-m/%-d/%y")
        predict_df = self.predict_mean(region_metadata, region_observations, run_day, train_start_date,
                                  train_end_date,
                                  search_space=search_space, is_tuning=True)
        metrics_result = evaluate_for_forecast(region_observations, predict_df, [loss_function])
        return metrics_result[0]["value"]

    def predict(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
                end_date: str, **kwargs):
        if self.model_parameters['modes']['predict_mode'] == 'with_uncertainty':
            return self.predict_with_uncertainty(region_metadata, region_observations, run_day, start_date, end_date)
        else:
            return self.predict_mean(region_metadata, region_observations, run_day, start_date, end_date, **kwargs)

    # @cached(cache=TTLCache(maxsize=2000, ttl=3600))
    def get_predictions_dict(self, region_metadata, region_observations, run_day, start_date, end_date):
        predictions_df_dict = dict()
        for idx in self.models:
            model = self.models[idx]
            predictions_df = model.predict(region_metadata, region_observations, run_day, start_date, end_date)
            predictions_df_dict[idx] = predictions_df.set_index("date")
        return predictions_df_dict

    def predict_mean(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
                     end_date: str, **kwargs):
        search_space = kwargs.get("search_space", {})
        self.model_parameters.update(search_space)
        beta = self.model_parameters['beta']
        predictions_df_dict = self.get_predictions_dict(region_metadata, region_observations,
                                                        run_day, start_date, end_date)
        self.weights = {idx: np.exp(-beta * loss) for idx, loss in self.losses.items()}
        predictions_df_dict = get_weighted_predictions(predictions_df_dict, self.weights)
        mean_predictions_df = reduce(lambda left, right: left.add(right), predictions_df_dict.values())
        mean_predictions_df = mean_predictions_df.div(sum(self.weights.values()))
        mean_predictions_df.reset_index(inplace=True)
        return mean_predictions_df

    def predict_with_uncertainty(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str,
                                 start_date: str, end_date: str, **kwargs):
        # refactor

        # Unpack uncertainty parameters
        uncertainty_params = self.model_parameters['uncertainty_parameters']
        date_of_interest = uncertainty_params['date_of_interest']
        column_of_interest = uncertainty_params['column_of_interest']
        include_mean = uncertainty_params['include_mean']
        percentiles = uncertainty_params['percentiles']
        ci = uncertainty_params['ci']  # multiple confidence intervals?
        alpha = 100 - ci
        confidence_intervals = {"low": alpha/2, "high": 100-alpha/2}
        tolerance = uncertainty_params['tolerance']

        percentiles_dict = dict()
        percentiles_forecast = dict()

        # Get predictions, mean predictions and weighted predictions
        predictions_df_dict = self.get_predictions_dict(region_metadata, region_observations,
                                                        run_day, start_date, end_date)
        mean_predictions_df = self.predict_mean(region_metadata, region_observations, run_day, start_date, end_date)

        # Get predictions on date of interest
        trials_df = create_trials_dataframe(predictions_df_dict, column_of_interest)
        try:
            predictions_doi = trials_df.loc[:, [date_of_interest]].reset_index(drop=True)
        except KeyError:
            raise Exception("The planning date is not in the range of predicted dates")
        df = pd.DataFrame.from_dict(self.weights, orient='index', columns=['weight'])
        df = df.join(predictions_doi.set_index(df.index))

        # Find PDF, CDF
        df['pdf'] = weights_to_pdf(df['weight'])
        df = df.sort_values(by=date_of_interest).reset_index()
        df['cdf'] = pdf_to_cdf(df['pdf'])

        # Get indices for percentiles and confidence intervals
        for p in percentiles:
            percentiles_dict[p] = get_best_index(df, p, tolerance)
        for c in confidence_intervals:
            percentiles_dict[c] = get_best_index(df, confidence_intervals[c], tolerance)

        # Create dictionary of dataframes for percentiles
        for key in percentiles_dict.keys():
            percentiles_forecast[key] = {}
            df_predictions = predictions_df_dict[percentiles_dict[key]]
            percentiles_forecast[key]['df_prediction'] = df_predictions

        percentiles_forecast = uncertainty_dict_to_df(percentiles_forecast)

        # Include mean predictions if include_mean is True
        if include_mean:
            # TODO: RESOLVE DATE TYPES AND USE A JOIN
            percentiles_forecast = pd.concat([mean_predictions_df, percentiles_forecast], axis=1)
            percentiles_forecast.drop(columns='predictionDate', inplace=True)

        return percentiles_forecast
