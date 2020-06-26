import pandas as pd
import numpy as np
from functools import reduce
import copy

from entities.forecast_variables import ForecastVariable
from model_wrappers.base import ModelWrapperBase
import model_wrappers.model_factory as model_factory_alias


class HomogeneousEnsemble(HeterogenousEnsemble):

    
    def get_mean_params(self):
        self.weights = {idx: np.exp(-beta * loss) for idx, loss in self.losses.items()}
        sum_of_weights = sum(self.weights.values())
        
        constituent_dict = self.model_parameters['constituent_models']
        model_class = ""
        mean_params = dict()
        for key in constituent_dict.keys():
            temp_model = constituent_dict[key]
            if(model_class = ""):
                model_class = temp_model['model_class']
            else:
                if(model_class != temp_model['model_class']):
                    print("Constituent Models not homogenous! Returning Null")
                    return None
            
            temp_params = constituent_dict[key]['model_parameters']
            for param_key in temp_params.keys():
                if(param_key in mean_params.keys()):
                    mean_params[keys] += (temp_params[param_key] * self.weights[key])/(sum_of_weights)
                else:
                    mean_params[keys] = (temp_params[param_key] * self.weights[key])/(sum_of_weights)
        
        return mean_params, model_class
    
    def predict_from_mean_param(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
                end_date: str, **kwargs):
        mean_params, model_class = self.get_mean_params()
        mean_param_model = model_factory_alias.ModelFactory.get_model(model_class, mean_params)
        
        prediction_df = mean_param_model.predict(region_metadata, region_observations, run_day, start_date, end_date)
        
        return prediction_df
        

#     def predict_with_uncertainty(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str,
#                                  start_date: str, end_date: str, uncertainty_params, **kwargs):
#         # refactor
#         date_of_interest = uncertainty_params['date_of_interest']
#         column_of_interest = uncertainty_params['column_of_interest']
#         include_mean = uncertainty_params['include_mean']
#         percentiles = uncertainty_params['percentiles']
#         # multiple confidence intervals?
#         ci = uncertainty_params['ci']
#         alpha = 100 - ci
#         confidence_intervals = {"low": alpha/2, "high": 100-alpha/2}

#         mean_predictions_df = self.predict(region_metadata, region_observations, run_day,
#                                            start_date, end_date, **kwargs)

#         predictions_df_list = self.get_weighted_predictions(region_metadata, region_observations, run_day,
#                                                             start_date, end_date)

#         trials_df = self.create_trials_dataframe(predictions_df_list, column_of_interest)
#         predictions_doi = trials_df.loc[:, [date_of_interest]].reset_index(drop=True)
#         df = pd.DataFrame.from_dict(self.weights, orient='index', columns=['weight'])
#         df = df.join(predictions_doi.set_index(df.index))

#         df['pdf'] = df['weight']/df['weight'].sum()
#         df = df.sort_values(by=date_of_interest)
#         df['cdf'] = df['pdf'].cumsum()

#         percentiles_dict = dict()

#         for p in percentiles:
#             idx = int((df['cdf'] - p/100).apply(abs).idxmin())
#             best_idx = df.iloc[idx - 2:idx + 2, :].index.min()
#             percentiles_dict[p] = int(best_idx)

#         for c in confidence_intervals:
#             idx = int((df['cdf'] - confidence_intervals[c] / 100).apply(abs).idxmin())
#             best_idx = df.iloc[idx - 2:idx + 2, :].index.min()
#             percentiles_dict[c] = int(best_idx)

#         percentiles_forecast = dict()

#         for key in percentiles_dict.keys():
#             percentiles_forecast[key] = {}
#             df_predictions = predictions_df_list[percentiles_dict[key]]
#             percentiles_forecast[key]['df_prediction'] = df_predictions

#         if include_mean:
#             percentiles_forecast['include_mean'] = {}
#             percentiles_forecast['include_mean']['df_prediction'] = mean_predictions_df

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