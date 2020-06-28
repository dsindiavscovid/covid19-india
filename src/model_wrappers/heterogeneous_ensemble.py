import pandas as pd
import numpy as np
from functools import reduce
import copy

from entities.forecast_variables import ForecastVariable
from model_wrappers.base import ModelWrapperBase
import model_wrappers.model_factory as model_factory_alias
from utils.ensemble_util import get_weighted_predictions, create_trials_dataframe, uncertainty_dict_to_df
from utils.distribution_util import weights_to_pdf, pdf_to_cdf, get_best_index


class HeterogeneousEnsemble(ModelWrapperBase):

    def __init__(self, model_parameters):
        self.model_parameters = model_parameters
        self.models = copy.copy(model_parameters['constituent_models'])
        self.losses = copy.copy(model_parameters['constituent_model_losses'])

        if 'constituent_model_weights' in model_parameters:
            self.weights = copy.copy(model_parameters['constituent_model_weights'])
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

    def get_latent_params(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, end_date: str,
                          search_space: dict):
        return dict()

    def get_predictions(self, region_metadata, region_observations, run_day, start_date, end_date):
        predictions_df_dict = dict()
        for idx in self.models:
            model = self.models[idx]
            predictions_df = model.predict(region_metadata, region_observations, run_day, start_date, end_date)
            predictions_df_dict[idx] = predictions_df.set_index("date")
        return predictions_df_dict

    def predict(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
                end_date: str, **kwargs):
        # what should the precomputed_predictions be named?
        search_space = kwargs.get("search_space", {})
        self.model_parameters.update(search_space)
        beta = self.model_parameters['beta']
        predictions_df_dict = kwargs.get("predictions", None)
        if predictions_df_dict is None:
            predictions_df_dict = self.get_predictions(region_metadata, region_observations,
                                                       run_day, start_date, end_date)
        self.weights = {idx: np.exp(-beta * loss) for idx, loss in self.losses.items()}
        predictions_df_dict = get_weighted_predictions(predictions_df_dict, self.weights)
        mean_predictions_df = reduce(lambda left, right: left.add(right), predictions_df_dict.values())
        mean_predictions_df = mean_predictions_df.div(sum(self.weights.values()))
        mean_predictions_df.reset_index(inplace=True)
        return mean_predictions_df

    def predict_with_uncertainty(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str,
                                 start_date: str, end_date: str, uncertainty_params, **kwargs):
        # refactor

        # Unpack uncertainty parameters
        date_of_interest = uncertainty_params['date_of_interest']
        column_of_interest = uncertainty_params['column_of_interest']
        include_mean = uncertainty_params['include_mean']
        percentiles = uncertainty_params['percentiles']
        ci = uncertainty_params['ci']  # multiple confidence intervals?
        alpha = 100 - ci
        confidence_intervals = {"low": alpha/2, "high": 100-alpha/2}
        window = uncertainty_params['window']

        percentiles_dict = dict()
        percentiles_forecast = dict()

        # Get predictions, mean predictions and weighted predictions
        predictions_df_dict = self.get_predictions(region_metadata, region_observations,
                                                   run_day, start_date, end_date)
        mean_predictions_df = self.predict(region_metadata, region_observations, run_day,
                                           start_date, end_date, predictions=predictions_df_dict, **kwargs)
        weighted_predictions_df_dict = get_weighted_predictions(predictions_df_dict, self.weights)

        # Get predictions on date of interest
        trials_df = create_trials_dataframe(weighted_predictions_df_dict, column_of_interest)
        predictions_doi = trials_df.loc[:, [date_of_interest]].reset_index(drop=True)
        df = pd.DataFrame.from_dict(self.weights, orient='index', columns=['weight'])
        df = df.join(predictions_doi.set_index(df.index))

        # Find PDF, CDF
        df['pdf'] = weights_to_pdf(df['weight'])
        df = df.sort_values(by=date_of_interest)
        df = df.reset_index()
        df['cdf'] = pdf_to_cdf(df['pdf'])

        # Get indices for percentiles and confidence intervals
        for p in percentiles:
            percentiles_dict[p] = get_best_index(df, p, window)
        for c in confidence_intervals:
            percentiles_dict[c] = get_best_index(df, confidence_intervals[c], window)

        # Create dictionary of dataframes for percentiles
        for key in percentiles_dict.keys():
            percentiles_forecast[key] = {}
            df_predictions = weighted_predictions_df_dict[percentiles_dict[key]]
            percentiles_forecast[key]['df_prediction'] = df_predictions

        # Include mean predictions if include_mean is True
        if include_mean:
            percentiles_forecast['include_mean'] = {}
            percentiles_forecast['include_mean']['df_prediction'] = mean_predictions_df

        # percentiles_forecast = self.uncertainty_dict_to_df(percentiles_forecast)
        return percentiles_forecast

    # def find_beta(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
    #               end_date: str, loss_function_config, **kwargs):
    #     max_evals = loss_function_config['max_evals']
    #     search_space = loss_function_config['search_space']
    #     training_loss_function = loss_function_config['training_loss_function']
    #     for k, v in search_space.items():
    #         search_space[k] = hp.uniform(k, v[0], v[1])
    #     objective = partial(self.optimize, region_metadata=region_metadata, region_observations=region_observations,
    #                         run_day=run_day, start_date=start_date, end_date=end_date,
    #                         loss_function=training_loss_function)
    #     # self.beta = hyperparam_tuning(objective, search_space, max_evals)
    #
    # def optimize(self, search_space, region_metadata, region_observations, run_day, start_date, end_date,
    #              loss_function):
    #     predict_df = self.predict(region_metadata, region_observations, run_day, start_date,
    #                               end_date, search_space=search_space)
    #     metrics_result = ModelEvaluator.evaluate_for_forecast(region_observations, predict_df, [loss_function])
    #     return metrics_result[0]["value"]
