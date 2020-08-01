from copy import deepcopy
from datetime import timedelta, datetime
from functools import reduce, partial

import numpy as np
import pandas as pd
from entities.forecast_variables import ForecastVariable
from entities.loss_function import LossFunction
from hyperopt import hp
from model_wrappers import model_factory as model_factory_alias
from model_wrappers.base import ModelWrapperBase
from utils.distribution_util import weights_to_pdf, pdf_to_cdf, get_best_index
from utils.ensemble_util import get_weighted_predictions, create_trials_dataframe, uncertainty_dict_to_df, get_weights
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

        # Initialize each constituent model
        self._initialize_constituent_models()

    def _initialize_constituent_models(self):
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
        run_day = (datetime.strptime(train_start_date, "%m/%d/%y") - timedelta(days=1)).strftime("%-m/%-d/%y")
        precomputed_pred = self.get_predictions_dict(region_metadata, region_observations, run_day,
                                                     train_start_date, train_end_date)
        if self.is_black_box():
            objective = partial(self.optimize, region_metadata=region_metadata, region_observations=region_observations,
                                train_start_date=train_start_date, train_end_date=train_end_date,
                                loss_function=train_loss_function, precomputed_pred=precomputed_pred)
            for k, v in search_space.items():
                search_space[k] = hp.uniform(k, v["low"], v["high"])
            result = hyperparam_tuning(objective, search_space,
                                       search_parameters.get("max_evals", 100))
        model_params = self.model_parameters
        model_params.update(result["best_params"])
        model_params["MAPE"] = result["best_loss"]
        result["model_parameters"] = model_params
        return {"model_parameters": model_params}

    def optimize(self, search_space, region_metadata, region_observations, train_start_date, train_end_date,
                 loss_function, precomputed_pred=None):
        run_day = (datetime.strptime(train_start_date, "%m/%d/%y") - timedelta(days=1)).strftime("%-m/%-d/%y")
        predict_df = self.predict_mean(region_metadata, region_observations, run_day, train_start_date,
                                       train_end_date,
                                       search_space=search_space, is_tuning=True,
                                       precomputed_pred=precomputed_pred)
        metrics_result = evaluate_for_forecast(region_observations, predict_df, [loss_function])
        return metrics_result[0]["value"]

    def predict(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
                end_date: str, **kwargs):
        if self.model_parameters['modes']['predict_mode'] == 'predictions_with_uncertainty':
            return self.predict_with_uncertainty(region_metadata, region_observations, run_day, start_date, end_date)
        elif self.model_parameters['modes']['predict_mode'] == 'best_fit':
            return self.predict_best_fit(region_metadata, region_observations, run_day, start_date, end_date)
        elif self.model_parameters['modes']['predict_mode'] == 'mean_predictions':
            return self.predict_mean(region_metadata, region_observations, run_day, start_date, end_date, **kwargs)
        else:
            raise Exception("Invalid Predict Mode")

    def get_predictions_dict(self, region_metadata, region_observations, run_day, start_date, end_date):
        """Gets predictions for all constituent models

        Args:
            region_metadata (dict): region metadata
            region_observations (pd.Dataframe): dataframe of case counts
            run_day (str): prediction run day
            start_date (str): prediction start date
            end_date (str): prediction end date

        Returns:
            dict(str:pd.DataFrame): dictionary of the form {index: predictions} for constituent models
        """

        predictions_df_dict = dict()
        for idx in self.models:
            model = self.models[idx]
            predictions_df = model.predict(region_metadata, region_observations, run_day, start_date, end_date)
            predictions_df_dict[idx] = predictions_df.set_index("date")
        return predictions_df_dict
    
    def predict_best_fit(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
                         end_date: str):
        """Gets predictions using constituent model with best fit on train period

        Args:
            region_metadata (dict): region metadata
            region_observations (pd.Dataframe): dataframe of case counts
            run_day (str): prediction run day
            start_date (str): prediction start date
            end_date (str): prediction end date

        Returns:
            pd.DataFrame: predictions
        """
        
        best_loss_idx = min(self.losses, key=self.losses.get)
        best_fit_model = self.models[best_loss_idx]
        return best_fit_model.predict(region_metadata, region_observations, run_day, start_date, end_date)

    def predict_mean(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
                     end_date: str, **kwargs):
        """Gets weighted mean predictions using constituent models

        Args:
            region_metadata (dict): region metadata
            region_observations (pd.DataFrame): dataframe of case counts
            run_day (str): prediction run day
            start_date (str): prediction start date
            end_date (str): prediction end date
            **kwargs: keyword arguments

        Returns:
            pd.DataFrame: mean predictions
        """

        search_space = kwargs.get("search_space", {})
        self.model_parameters.update(search_space)
        beta = self.model_parameters['beta']
        precomputed_pred = kwargs.get("precomputed_pred", None)
        is_tuning = kwargs.get("is_tuning", False)

        # Get predictions from constituent models
        if precomputed_pred is None:
            predictions_df_dict = self.get_predictions_dict(region_metadata, region_observations,
                                                            run_day, start_date, end_date)
        else:
            predictions_df_dict = deepcopy(precomputed_pred)

        # Calculate weights for constituent models as exp(-beta*loss)
        if self.weights is None or is_tuning:
            weights = get_weights(beta, self.losses)
        else:
            weights = deepcopy(self.weights)
       
        # Get weighted predictions of constituent models
        predictions_df_dict = get_weighted_predictions(predictions_df_dict, weights)

        # Compute mean predictions
        mean_predictions_df = reduce(lambda left, right: left.add(right), predictions_df_dict.values())
        if sum(weights.values()) != 0:
            mean_predictions_df = mean_predictions_df.div(sum(weights.values()))
        mean_predictions_df.reset_index(inplace=True)
    
        return mean_predictions_df
    
    # Helper function to get the model indexes
    def _get_index_for_percentile_helper(self, variable_of_interest, date_of_interest, tolerance, percentiles,
                                         region_metadata, region_observations, run_day, start_date, end_date):

        # Get predictions, mean predictions and weighted predictions
        predictions_df_dict = self.get_predictions_dict(region_metadata, region_observations,
                                                        run_day, start_date, end_date)

        # Get predictions for a specific column on date of interest
        trials_df = create_trials_dataframe(predictions_df_dict, variable_of_interest)
        try:
            predictions_doi = trials_df.loc[:, [date_of_interest]].reset_index(drop=True)
        except KeyError:
            raise Exception("The planning date is not in the range of predicted dates")
        beta = self.model_parameters['beta']
        if self.weights is None:
            weights = get_weights(beta, self.losses)
        else:
            weights = deepcopy(self.weights)
        df = pd.DataFrame.from_dict(weights, orient='index', columns=['weight'])
        df = df.join(predictions_doi.set_index(df.index))

        # Find PDF, CDF
        df['pdf'] = weights_to_pdf(df['weight'])
        df = df.sort_values(by=date_of_interest).reset_index()
        df['cdf'] = pdf_to_cdf(df['pdf'])

        # Get indices for percentiles and confidence intervals
        percentiles_dict = dict()
        for p in percentiles:
            percentiles_dict[p] = get_best_index(df, p, tolerance)
        return percentiles_dict, predictions_df_dict

    def predict_with_uncertainty(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str,
                                 start_date: str, end_date: str, **kwargs):
        """Get predictions for percentiles and confidence intervals, and (optionally) mean predictions

        Args:
            region_metadata (dict): region metadata
            region_observations (pd.DataFrame): dataframe of case counts
            run_day (str): prediction run day
            start_date (str): prediction start date
            end_date (str): prediction end date
            **kwargs: keyword arguments

        Returns:
            pd.DataFrame: dataframe with predictions for percentiles and confidence intervals,
                and optionally mean predictions
        """

        # Unpack uncertainty parameters
        uncertainty_params = self.model_parameters['uncertainty_parameters']
        date_of_interest = uncertainty_params['date_of_interest']
        variable_of_interest = uncertainty_params['variable_of_interest']
        include_mean = uncertainty_params['include_mean']
        percentiles = uncertainty_params['percentiles']
        ci = uncertainty_params['confidence_interval_sizes']
        confidence_intervals = []
        for c in ci:
            confidence_intervals.extend([50 - c/2, 50 + c/2])
        tolerance = uncertainty_params['tolerance']

        percentiles_dict, predictions_df_dict = self._get_index_for_percentile_helper(
            variable_of_interest, date_of_interest, tolerance, percentiles+confidence_intervals,
            region_metadata, region_observations, run_day, start_date, end_date)

        # Create dictionary of dataframes for percentiles
        percentiles_predictions = dict()
        for key in percentiles_dict.keys():
            percentiles_predictions[key] = {}
            df_predictions = predictions_df_dict[percentiles_dict[key]]
            percentiles_predictions[key]['df_prediction'] = df_predictions

        # Create percentiles dataframe
        percentiles_predictions = uncertainty_dict_to_df(percentiles_predictions)

        # Include mean predictions in dataframe if include_mean is True
        if include_mean:
            mean_predictions_df = self.predict_mean(region_metadata, region_observations, run_day, start_date, end_date)
            # TODO: RESOLVE DATE TYPES AND USE A JOIN INSTEAD OF CONCAT
            percentiles_predictions = pd.concat([mean_predictions_df, percentiles_predictions], axis=1)
            percentiles_predictions.drop(columns='predictionDate', inplace=True)

        return percentiles_predictions
    
    def get_params_for_percentiles(self, variable_of_interest, date_of_interest, tolerance, percentiles,
                                   region_metadata, region_observations, run_day, start_date, end_date):
        
        percentiles_dict, _ = self._get_index_for_percentile_helper(variable_of_interest, date_of_interest, tolerance,
                                                                    percentiles, region_metadata, region_observations, 
                                                                    run_day, start_date, end_date)
        return_dict = dict()
        for decile in percentiles_dict.keys():
            return_dict[decile] = self.models[percentiles_dict[decile]].model_parameters
    
        return return_dict

    def get_trials_distribution(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str,
                                start_date: str, end_date: str):
        """Get trials and distribution for a given column and date

        Args:
            region_metadata (dict): region metadata
            region_observations (pd.DataFrame): dataframe of case counts
            run_day (str): prediction run day
            start_date (str): prediction start date
            end_date (str): prediction end date

        Returns:
            pd.DataFrame: dataframe with columns ['case_counts', 'weight', 'pdf', 'cdf']
        """
        # Unpack uncertainty parameters
        uncertainty_params = self.model_parameters['uncertainty_parameters']
        date_of_interest = uncertainty_params['date_of_interest']
        variable_of_interest = uncertainty_params['variable_of_interest']

        # Get predictions, mean predictions and weighted predictions
        predictions_df_dict = self.get_predictions_dict(region_metadata, region_observations,
                                                        run_day, start_date, end_date)

        # Get predictions for a specific column on date of interest
        trials_df = create_trials_dataframe(predictions_df_dict, variable_of_interest)
        try:
            predictions_doi = trials_df.loc[:, [date_of_interest]].reset_index(drop=True)
        except KeyError:
            raise Exception("The planning date is not in the range of predicted dates")
        beta = self.model_parameters['beta']
        if self.weights is None:
            weights = get_weights(beta, self.losses)
        else:
            weights = deepcopy(self.weights)
        df = pd.DataFrame.from_dict(weights, orient='index', columns=['weight'])
        df = df.join(predictions_doi.set_index(df.index))
        df = df.rename(columns={date_of_interest: 'case_counts'})

        # Find PDF, CDF
        df['pdf'] = weights_to_pdf(df['weight'])
        df = df.sort_values(by='case_counts').reset_index()
        df['cdf'] = pdf_to_cdf(df['pdf'])

        return df
