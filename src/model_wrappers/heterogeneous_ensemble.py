import pandas as pd
import numpy as np
from functools import reduce
import copy

from entities.forecast_variables import ForecastVariable
from model_wrappers.base import ModelWrapperBase
import model_wrappers.model_factory as model_factory_alias


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

    def get_weighted_predictions(self, region_metadata, region_observations, run_day, start_date, end_date):

        predictions_df_list = []

        for idx in self.models:
            model = self.models[idx]
            predictions_df = model.predict(region_metadata, region_observations, run_day, start_date, end_date)
            predictions_df = predictions_df.set_index("date").multiply(self.weights[idx])
            predictions_df_list.append(predictions_df)

        return predictions_df_list

    def predict(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
                end_date: str, **kwargs):

        search_space = kwargs.get("search_space", {})
        self.model_parameters.update(search_space)
        beta = self.model_parameters['beta']

        self.weights = {idx: np.exp(-beta * loss) for idx, loss in self.losses.items()}

        predictions_df_list = self.get_weighted_predictions(region_metadata, region_observations, run_day,
                                                            start_date, end_date)

        mean_predictions_df = reduce(lambda left, right: left.add(right), predictions_df_list)
        mean_predictions_df = mean_predictions_df.div(sum(self.weights.values()))
        mean_predictions_df.reset_index(inplace=True)

        return mean_predictions_df

    def create_trials_dataframe(self, predictions_df_list, column=ForecastVariable.active):
        # move to utils
        trials_df_list = []
        for i in range(len(predictions_df_list)):
            trials_df_list.append(predictions_df_list[i].loc[:, [column]].T)
        return pd.concat(trials_df_list, axis=0, ignore_index=True)

    def predict_with_uncertainty(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str,
                                 start_date: str, end_date: str, uncertainty_params, **kwargs):
        # refactor
        date_of_interest = uncertainty_params['date_of_interest']
        column_of_interest = uncertainty_params['column_of_interest']
        include_mean = uncertainty_params['include_mean']
        percentiles = uncertainty_params['percentiles']
        # multiple confidence intervals?
        ci = uncertainty_params['ci']
        alpha = 100 - ci
        confidence_intervals = {"low": alpha/2, "high": 100-alpha/2}

        mean_predictions_df = self.predict(region_metadata, region_observations, run_day,
                                           start_date, end_date, **kwargs)

        predictions_df_list = self.get_weighted_predictions(region_metadata, region_observations, run_day,
                                                            start_date, end_date)

        trials_df = self.create_trials_dataframe(predictions_df_list, column_of_interest)
        predictions_doi = trials_df.loc[:, [date_of_interest]].reset_index(drop=True)
        df = pd.DataFrame.from_dict(self.weights, orient='index', columns=['weight'])
        df = df.join(predictions_doi.set_index(df.index))

        df['pdf'] = df['weight']/df['weight'].sum()
        df = df.sort_values(by=date_of_interest)
        df['cdf'] = df['pdf'].cumsum()

        percentiles_dict = dict()

        for p in percentiles:
            idx = int((df['cdf'] - p/100).apply(abs).idxmin())
            best_idx = df.iloc[idx - 2:idx + 2, :].index.min()
            percentiles_dict[p] = int(best_idx)

        for c in confidence_intervals:
            idx = int((df['cdf'] - confidence_intervals[c] / 100).apply(abs).idxmin())
            best_idx = df.iloc[idx - 2:idx + 2, :].index.min()
            percentiles_dict[c] = int(best_idx)

        percentiles_forecast = dict()

        for key in percentiles_dict.keys():
            percentiles_forecast[key] = {}
            df_predictions = predictions_df_list[percentiles_dict[key]]
            percentiles_forecast[key]['df_prediction'] = df_predictions

        if include_mean:
            percentiles_forecast['include_mean'] = {}
            percentiles_forecast['include_mean']['df_prediction'] = mean_predictions_df

        # Convert to dataframe

        return percentiles_forecast

    # def find_beta(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
    #               end_date: str, loss_function_config, **kwargs):
    #     # def find_beta(self, region_metadata, region_observations, train_start_date, train_end_date, search_space,
    #     #               search_parameters, train_loss_function):
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
