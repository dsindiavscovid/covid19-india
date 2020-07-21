import pandas as pd
import numpy as np

import copy
from datetime import timedelta, datetime
from functools import partial
from hyperopt import hp

from entities.forecast_variables import ForecastVariable
from entities.loss_function import LossFunction
from entities.model_class import ModelClass
from model_wrappers import model_factory as model_factory_alias
from utils.ensemble_util import uncertainty_dict_to_df
from utils.distribution_util import weights_to_pdf, pdf_to_cdf, get_best_index
from utils.hyperparam_util import hyperparam_tuning_ensemble
from model_wrappers.heterogeneous_ensemble import HeterogeneousEnsemble

class HomogeneousEnsemble(HeterogeneousEnsemble):
    
    def __init__(self, model_parameters):
        self.child_model_class = ModelClass(model_parameters['child_model']['model_class'])
        self.child_model_parameters = model_parameters['child_model']['model_parameters']
        
        self.model_parameters = model_parameters
        if('constituent_models' in model_parameters.keys()):
            super().__init__(model_parameters)
            for key in model_parameters['constituent_models']:
                temp_model = model_parameters['constituent_models'][key]
                if(self.child_model_class != temp_model['model_class']):
                    raise Exception("Constituent Models not homogenous! Model Class differs.")
    
    def supported_forecast_variables(self):
        return [ForecastVariable.confirmed, ForecastVariable.recovered, ForecastVariable.active]
    
    def update_nested_dict(self, meandict, x, key):
        if(type(x) == dict):
            if(key not in meandict.keys()):
                meandict[key] = dict()
            for k in x.keys():
                meandict[key] = self.update_nested_dict(meandict[key], x[k], k)
            return meandict
        else:
            if(key in meandict.keys()):
                meandict[key] += x
            else:
                meandict[key] = x
            return meandict
    
    def get_mean_params(self):
        if(self.weights == None):
            self.weights = {idx: np.exp(-self.model_parameters['beta'] * loss) for idx, loss in self.losses.items()}

        constituent_dict = self.model_parameters['constituent_models']
        mean_params = dict()
        for idx in constituent_dict.keys():
            constituent_model = constituent_dict[idx]['model_parameters']
            for key in constituent_model:
                mean_params.update(self.update_nested_dict(mean_params, constituent_model[key], key))
        return mean_params
    
    def get_params_uncertainty(self): 
        uncertainty_params = self.model_parameters['uncertainty_parameters']
        param_of_interest = uncertainty_params['param_key_of_interest']
        include_mean = uncertainty_params['include_mean']
        percentiles = uncertainty_params['percentiles']
        ci = uncertainty_params['ci']  # multiple confidence intervals?
        alpha = 100 - ci
        confidence_intervals = {"low": alpha/2, "high": 100-alpha/2}
        window = uncertainty_params['window']
        
        beta = float(self.model_parameters['beta'])
        if self.weights is None:
            weights = {idx: np.exp(-beta * loss) for idx, loss in self.losses.items()}
        else:
            weights = deepcopy(self.weights)
            
        params_dict = dict()
        for idx in self.model_parameters['constituent_model_losses']:
            params_dict[idx] = [weights[idx], 
                                self.model_parameters['constituent_models'][idx]['model_parameters'][param_of_interest]]
    
        params_df = pd.DataFrame.from_dict(params_dict, orient = 'index', columns = ['weight', 'ParamOfInterest'])
        
        params_df['pdf'] = weights_to_pdf(params_df['weight'])
        params_df = params_df.sort_values(by='ParamOfInterest').reset_index()
        params_df['cdf'] = pdf_to_cdf(params_df['pdf'])

        percentiles_dict = dict()
        for p in percentiles:
            percentiles_dict[p] = get_best_index(params_df, p, window)
        for c in confidence_intervals:
            percentiles_dict[c] = get_best_index(params_df, confidence_intervals[c], window)
            
        percentiles_params = dict()
        for key in percentiles_dict.keys():
            percentiles_params[key] = {}
            idx = percentiles_dict[key]
            percentiles_params[key]['model_parameters'] = self.model_parameters['constituent_models'][idx]['model_parameters']
            percentiles_params[key]['model_index'] = idx
            
        if(include_mean):
            mean_params = self.get_mean_params()
            percentiles_params['mean'] = dict()
            percentiles_params['mean']['model_parameters'] = mean_params
        
        return percentiles_params
    
    def _flatten_dict(self, oldDict, appendStr = ""):
        newDict = dict()
        for key in oldDict.keys():
            if(type(oldDict[key]) ==  dict):
                newDict.update(self._flatten_dict(oldDict[key], appendStr+key+"_"))
            else:
                if(isinstance(oldDict[key], float) or isinstance(oldDict[key], int)):
                    newDict[appendStr+key] = oldDict[key]
        return newDict
        
    def _get_statistics_given_indexes(self, list_of_indexes, appendStr):
        if(len(list_of_indexes)<=0):
            return pd.DataFrame()
        list_of_params = dict()
        for idx in list_of_indexes:
            list_of_params[idx] = (self._flatten_dict(self.models[idx].model_parameters))
        
        keys = list(list_of_params[list_of_indexes[0]].keys())
        
        beta = float(self.model_parameters['beta'])
        if self.weights is None:
            weights = {idx: np.exp(-beta * loss) for idx, loss in self.losses.items()}
        else:
            weights = deepcopy(self.weights)
            
        s = sum(weights.values())
        for idx in weights.keys():
            weights[idx] = weights[idx]/s
            
        mean = dict()
        mini = dict()
        maxi = dict()
        mean['statName'] = appendStr+"Mean"
        mini['statName'] = appendStr+"Min"
        maxi['statName'] = appendStr+"Max"
        for key in keys:
            mean[key] = np.sum([list_of_params[idx][key]*weights[idx] for idx in list_of_indexes])/sum(weights.values())
            mini[key] = np.min([list_of_params[idx][key] for idx in list_of_indexes])
            maxi[key] = np.max([list_of_params[idx][key] for idx in list_of_indexes])
            
        return [mean, mini, maxi]

    
    def get_statistics_of_params(self, output_file_location = None):
        sortedLossIndexes =[item[0] for item in sorted(self.losses.items(), key = lambda kv:(kv[1], kv[0]))]
        
        statsList = []
        
        statsList.extend(self._get_statistics_given_indexes(sortedLossIndexes[:10], "top10_"))
        statsList.extend(self._get_statistics_given_indexes(sortedLossIndexes[:50], "top50_"))
        statsList.extend(self._get_statistics_given_indexes(sortedLossIndexes, "all_"))
        statsDF = pd.DataFrame(columns = list(statsList[0].keys()))
        for s in statsList:
            statsDF = statsDF.append(s, ignore_index=True)
            
        if output_file_location is not None:
            statsDF.to_csv(output_file_location)
        return statsDF
        
        
    
    def predict_from_mean_param(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
                end_date: str, **kwargs):
        mean_params = self.get_mean_params()
        mean_param_model = model_factory_alias.ModelFactory.get_model(self.child_model_class, mean_params)
        prediction_df = mean_param_model.predict(region_metadata, region_observations, run_day, start_date, end_date)
        return prediction_df
          
    def get_predictions_dict_some_indexes(self, region_metadata, region_observations, run_day, start_date, end_date, indexes):
        predictions_df_dict = dict()
        for idx in indexes:
            model = self.models[idx]
            predictions_df = model.predict(region_metadata, region_observations, run_day, start_date, end_date)
            predictions_df_dict[idx] = predictions_df.set_index("date")
        return predictions_df_dict
    
   
    def predict_with_decile_uncertainty(self, region_metadata: dict, region_observations: pd.DataFrame, 
                                        run_day: str, start_date: str, end_date: str):
        
        percentile_params = self.get_params_uncertainty()
        indexes = []
        for key in percentile_params.keys():
            if(key == "mean"):
                continue
            indexes.append(percentile_params[key]['model_index'])
            
        predictions_df_dict = self.get_predictions_dict_some_indexes(region_metadata, region_observations,
                                                        run_day, start_date, end_date, indexes)
        
        percentiles_forecast = dict()
        for key in percentile_params.keys():
            if(key == "mean"):
                continue
            percentiles_forecast[key] = dict()
            df_predictions = predictions_df_dict[percentile_params[key]['model_index']]
            percentiles_forecast[key]['df_prediction'] = df_predictions
       
        percentiles_forecast = uncertainty_dict_to_df(percentiles_forecast)
        if(self.model_parameters['uncertainty_parameters']['include_mean']):
            mean_predictions_df = self.predict_from_mean_param(region_metadata, region_observations, run_day, start_date, end_date)
            percentiles_forecast = pd.concat([mean_predictions_df, percentiles_forecast], axis=1)
        return percentiles_forecast
    
    
        
    def train_for_ensemble(self, region_metadata, region_observations, train_start_date, train_end_date, search_space,
                           search_parameters, train_loss_function):
        child_model = model_factory_alias.ModelFactory.get_model(self.child_model_class, self.child_model_parameters)
        if child_model.is_black_box():
            objective = partial(child_model.optimize, region_metadata=region_metadata, region_observations=region_observations,
                                train_start_date=train_start_date,
                                train_end_date=train_end_date, loss_function=train_loss_function)
            for k, v in search_space.items():
                search_space[k] = hp.uniform(k, v[0], v[1])
            result_list = hyperparam_tuning_ensemble(objective, search_space,
                                                     search_parameters.get("max_evals", 100))
            run_day = (datetime.strptime(train_start_date, "%m/%d/%y") - timedelta(days=1)).strftime(
                "%-m/%-d/%y")
            constituent_models = dict()
            constituent_model_losses = dict()
            for i in range(len(result_list)):
                result = result_list[i]
                model_params = copy.deepcopy(self.child_model_parameters)
                model_params.update(result[0]) 
                latent_params = child_model.get_latent_params(region_metadata, region_observations, run_day,
                                                              train_end_date, model_params)
                model_params.update(latent_params["latent_params"])
                tempDict = dict()
                tempDict['model_class'] = self.child_model_class.name
                tempDict['model_parameters'] = model_params
                constituent_models[str(i)] = tempDict
                constituent_model_losses[str(i)] = result[1]

        self.model_parameters.update({"constituent_models": constituent_models, "constituent_model_losses": constituent_model_losses})

        return {"model_parameters": self.model_parameters}
    
    def train(self, region_metadata: dict, region_observations: pd.DataFrame, train_start_date: str,
              train_end_date: str, search_space: dict, search_parameters: dict, train_loss_function: LossFunction):
        
        if(self.model_parameters['modes']['training_mode'] == 'only_beta'):
            onlyBetaSearchSpace = dict()
            onlyBetaSearchSpace['beta'] = search_space['beta']
            return super().train(region_metadata, region_observations, train_start_date,
              train_end_date, onlyBetaSearchSpace, search_parameters, train_loss_function)
        elif(self.model_parameters['modes']['training_mode'] == 'constituent_models'):
            if('beta' in search_space.keys()):
                search_space.pop('beta')
            return self.train_for_ensemble(region_metadata, region_observations, train_start_date,
              train_end_date, search_space, search_parameters, train_loss_function)
        elif(self.model_parameters['modes']['training_mode'] == 'full'):
            onlyBetaSearchSpace = dict()
            onlyBetaSearchSpace['beta'] = copy.deepcopy(search_space['beta'])
            search_space.pop('beta')
            ndays = (datetime.strptime(train_end_date, "%m/%d/%y") - (datetime.strptime(train_start_date, "%m/%d/%y"))).days

            train1_end_date = (datetime.strptime(train_start_date, "%m/%d/%y") + timedelta(days=ndays * float(search_parameters['frac_for_child']))).strftime("%-m/%-d/%y")
            self.train_for_ensemble(region_metadata, region_observations, train_start_date,
              train1_end_date, search_space, search_parameters["child_model"], train_loss_function)

            self.models = {k: v for k, v in self.model_parameters['constituent_models'].items() if
                           int(k) < self.model_parameters["n"]}
            self.losses = {k: v for k, v in self.model_parameters['constituent_model_losses'].items() if
                           int(k) < self.model_parameters["n"]}

            if 'constituent_model_weights' in self.model_parameters:
                self.weights = copy.deepcopy(self.model_parameters['constituent_model_weights'])
            else:
                self.weights = None

            for idx in self.models:
                constituent_model = self.models[idx]
                constituent_model_class = constituent_model['model_class']
                constituent_model_parameters = constituent_model['model_parameters']
                self.models[idx] = model_factory_alias.ModelFactory.get_model(
                    constituent_model_class, constituent_model_parameters)

            resultsBeta = super().train(region_metadata, region_observations, train_start_date,
              train_end_date, onlyBetaSearchSpace, search_parameters["ensemble_model"], train_loss_function)
            self.model_parameters.update(resultsBeta['model_parameters'])
            return {"model_parameters": self.model_parameters}
            
            
    def predict(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
                end_date: str, **kwargs):
        
        
        if(self.model_parameters['modes']['predict_mode'] == 'predictions_with_uncertainty'):
            return super().predict_with_uncertainty(region_metadata, region_observations, run_day, start_date, end_date)
           
        elif(self.model_parameters['modes']['predict_mode'] == 'params_with_uncertainty'):
            return self.predict_with_decile_uncertainty(region_metadata, region_observations, run_day, start_date, end_date)
             
        elif(self.model_parameters['modes']['predict_mode'] == 'mean_predictions'):
            return super().predict_mean(region_metadata, region_observations, run_day, start_date, end_date)
             
        elif(self.model_parameters['modes']['predict_mode'] == 'mean_params'):
            return self.predict_from_mean_param(region_metadata, region_observations, run_day, start_date, end_date)
        
        elif(self.model_parameters['modes']['predict_mode'] == 'best_fit'):
            return super().predict_best_fit(region_metadata, region_observations, run_day, start_date, end_date)
        
        else:
             raise Exception("Invalid Predict Mode")
