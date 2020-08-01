import copy
from datetime import timedelta, datetime
from functools import reduce, partial

import pandas as pd
from entities.forecast_variables import ForecastVariable
from entities.loss_function import LossFunction
from hyperopt import hp
from model_wrappers.base import ModelWrapperBase
from seirsplus.models_SEIHRD_gen import SEIHRDModel
from utils.hyperparam_util import hyperparam_tuning
from utils.loss_util import evaluate_for_forecast


class SEIHRD_gen(ModelWrapperBase):

    def fit(self):
        pass

    def __init__(self, model_parameters: dict):
        self.model_parameters = copy.deepcopy(model_parameters)
        # Not using Pydantic Method because in the training output we want string returning instead of entity objects
        if "latent_information" in self.model_parameters.keys():
            if "latent_variables" in self.model_parameters["latent_information"].keys():
                self.model_parameters["latent_information"]["latent_variables"] = [ForecastVariable(fv) for fv in
                                                                                   self.model_parameters[
                                                                                       "latent_information"][
                                                                                       "latent_variables"]]

            if "latent_on" in self.model_parameters["latent_information"].keys():
                self.model_parameters["latent_information"]["latent_on"] = ForecastVariable(
                    self.model_parameters["latent_information"]["latent_on"])

    def _convertLatentStrToForecastEntity(self, model_parameters):
        if "latent_information" in model_parameters.keys():
            if "latent_variables" in model_parameters["latent_information"].keys():
                model_parameters["latent_information"]["latent_variables"] = [ForecastVariable(fv) for fv in
                                                                              model_parameters["latent_information"][
                                                                                  "latent_variables"]]

            if "latent_on" in model_parameters["latent_information"].keys():
                model_parameters["latent_information"]["latent_on"] = ForecastVariable(
                    model_parameters["latent_information"]["latent_on"])
        return model_parameters

    def supported_forecast_variables(self):
        return [ForecastVariable.active, ForecastVariable.exposed, ForecastVariable.hospitalized,
                ForecastVariable.recovered, ForecastVariable.deceased]

    def input_variables(self):
        return ForecastVariable.input_variables()

    def optimize(self, search_space, region_metadata, region_observations, train_start_date, train_end_date,
                 loss_function):
        run_day = (datetime.strptime(train_start_date, "%m/%d/%y") - timedelta(days=1)).strftime("%-m/%-d/%y")
        predict_df = self.predict(region_metadata, region_observations, run_day, train_start_date,
                                  train_end_date, search_space=search_space, is_tuning=True)
        metrics_result = evaluate_for_forecast(region_observations, predict_df, [loss_function])
        return metrics_result[0]["value"]

    def train(self, region_metadata: dict, region_observations: pd.DataFrame, train_start_date: str,
              train_end_date: str, search_space: dict, search_parameters: dict, train_loss_function: LossFunction):
        result = {}
        if self.is_black_box():
            run_day = (datetime.strptime(train_start_date, "%m/%d/%y") - timedelta(days=1)).strftime(
                "%-m/%-d/%y")
            objective = partial(self.optimize, region_metadata=region_metadata, region_observations=region_observations,
                                train_start_date=train_start_date, train_end_date=train_end_date,
                                loss_function=train_loss_function)
            for k, v in search_space.items():
                search_space[k] = hp.uniform(k, v["low"], v["high"])
            result = hyperparam_tuning(objective, search_space,
                                       search_parameters.get("max_evals", 100))
            latent_params = self.get_latent_params(region_metadata, region_observations, run_day,
                                                   train_end_date, result["best_params"])
            result.update(latent_params)

        model_params = self.model_parameters
        model_params.update(latent_params["latent_params"])
        model_params.update(result["best_params"])
        model_params["MAPE"] = result["best_loss"]
        result["model_parameters"] = model_params
        return {"model_parameters": model_params}

    def predict(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
                end_date: str, **kwargs):
        latent_variables = self.model_parameters['latent_information']['latent_variables']
        latent_on = self.model_parameters['latent_information']['latent_on']
        search_space = kwargs.get("search_space", {})
        self._is_tuning = kwargs.get("is_tuning", False)
        self.model_parameters.update(search_space)
        self.model_parameters = self._convertLatentStrToForecastEntity(copy.deepcopy(self.model_parameters))
        n_days = (datetime.strptime(end_date, "%m/%d/%y") - datetime.strptime(run_day, "%m/%d/%y")).days + 1
        prediction_dataset = self.run(region_observations, region_metadata, run_day, n_days, latent_variables,
                                      latent_on)
        date_list = list(pd.date_range(start=start_date, end=end_date).strftime("%-m/%-d/%y"))
        prediction_dataset = prediction_dataset[prediction_dataset.date.isin(date_list)]
        return prediction_dataset

    def is_black_box(self):
        return True

    def get_latent_params(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, end_date: str,
                          search_space: dict = {}, latent_variables: list = [],
                          latent_on: ForecastVariable = ForecastVariable.confirmed):
        latent_variables = self.model_parameters['latent_information']['latent_variables']
        latent_on = self.model_parameters['latent_information']['latent_on']
        self.model_parameters.update(search_space)
        self.model_parameters = self._convertLatentStrToForecastEntity(copy.deepcopy(self.model_parameters))
        n_days = (datetime.strptime(end_date, "%m/%d/%y") - datetime.strptime(run_day, "%m/%d/%y")).days + 1
        prediction_dataset = self.run(region_observations, region_metadata, run_day, n_days, latent_variables,
                                      latent_on)
        params = dict()
        params['latent_params'] = dict()
        ed = prediction_dataset[prediction_dataset['date'] == end_date]
        rd = prediction_dataset[prediction_dataset['date'] == run_day]
        for latent_var in latent_variables:
            params['latent_params']["Latent_" + latent_var.name + "_ratio"] = dict()
            params['latent_params']["Latent_" + latent_var.name + "_ratio"][run_day] = float(
                rd[latent_var.name]) / float(
                rd[latent_on.name])
            params['latent_params']["Latent_" + latent_var.name + "_ratio"][end_date] = float(
                ed[latent_var.name]) / float(
                ed[latent_on.name])
        return params

    def getOutsideToModelMap(self):
        # Should cover all the variables required by model as input
        d = dict()
        d['initI'] = ForecastVariable.active.name
        d['initE'] = ForecastVariable.exposed.name
        d['initH'] = ForecastVariable.hospitalized.name
        d['initR'] = ForecastVariable.recovered.name
        d['initD'] = ForecastVariable.deceased.name
        return d

    def getModelToOutside(self, estimator):
        # Should cover all the variables returned by supported_forecast_variables
        d = dict()
        d[ForecastVariable.exposed.name] = estimator.numE
        d[ForecastVariable.active.name] = estimator.numI
        d[ForecastVariable.hospitalized.name] = [estimator.numHr[i] + estimator.numHd[i] for i in
                                                 range(len(estimator.numHr))]
        d[ForecastVariable.recovered.name] = estimator.numR
        d[ForecastVariable.deceased.name] = estimator.numD
        return d

    def get_model_with_params(self, region_metadata, region_observations, run_day,
                              latent_variables: list = [],
                              latent_on: ForecastVariable = ForecastVariable.confirmed):
        r0 = self.model_parameters['r0']
        init_sigma = 1. / self.model_parameters['incubation_period']
        init_beta = r0 * init_sigma
        init_gamma = 1. / self.model_parameters['infectious_period']
        init_alpha = 1. / self.model_parameters['recovery_period']
        init_delta = 1. / self.model_parameters['deceased_period']
        init_kappa = self.model_parameters['recovered_ratio']
        initN = region_metadata.get("population")

        datasets = dict()
        initDict = dict()

        for var in self.input_variables():
            temp_dataset = region_observations[region_observations.observation == var.name].iloc[0]
            datasets[var.name] = temp_dataset

        if self._is_tuning:
            for var in latent_variables:
                initDict[var.name] = datasets[latent_on.name][run_day] * self.model_parameters.get(var.name + '_ratio')

        else:
            if run_day not in self.model_parameters.get("Latent_{}_ratio".format(latent_variables[0].name)):
                """
                When model_parameters don't have latent_params for run day, 
                we run the model from the most recent day for which we have latent_params till the run_day 
                
                """

                latent_days = list(self.model_parameters.get("Latent_{}_ratio".format(latent_variables[0].name)).keys())
                temp_run_day = run_day
                while not temp_run_day in latent_days:
                    temp_run_day = (datetime.strptime(temp_run_day, "%m/%d/%y") - timedelta(days=1)).strftime(
                        "%-m/%-d/%y")

                new_latent_params = self.get_latent_params(region_metadata, region_observations, temp_run_day, run_day,
                                                           latent_variables=latent_variables, latent_on=latent_on)[
                    'latent_params']

                for latent_key in new_latent_params:
                    self.model_parameters[latent_key].update(new_latent_params[latent_key])

            for var in latent_variables:
                initDict[var.name] = datasets[latent_on.name][run_day] * self.model_parameters.get(
                    'Latent_{}_ratio'.format(var.name)).get(run_day)

        for var in self.input_variables():
            initDict[var.name] = datasets[var.name][run_day]

        oToM = self.getOutsideToModelMap()
        estimator = SEIHRDModel(beta=init_beta, sigma=init_sigma, gamma=init_gamma, alpha=init_alpha, delta=init_delta,
                                kappa=init_kappa, initN=initN, initI=initDict[oToM['initI']],
                                initE=initDict[oToM['initE']],
                                initHr=(initDict[oToM['initH']] * init_kappa),
                                initHd=(initDict[oToM['initH']] * (1 - init_kappa)),
                                initR=initDict[oToM['initR']], initD=initDict[oToM['initD']])

        return estimator

    def run(self, region_observations: pd.DataFrame, region_metadata, run_day: str, n_days: int,
            latent_variables: list, latent_on: ForecastVariable):
        estimator = self.get_model_with_params(region_metadata, region_observations, run_day, latent_variables,
                                               latent_on)
        estimator.run(T=n_days, verbose=False)

        mToO = self.getModelToOutside(estimator)
        data_frames = []
        for var in self.supported_forecast_variables():
            data_frames.append(self.alignTimeSeries(mToO[var.name], estimator.tseries, run_day, n_days, var.name))

        data_frames.append(self.alignTimeSeries(
            [sum(x) for x in zip(estimator.numHr, estimator.numHd, estimator.numR, estimator.numD)], estimator.tseries,
            run_day, n_days, ForecastVariable.confirmed.name))

        result = reduce(lambda left, right: pd.merge(left, right, on=['date'], how='inner'), data_frames)
        result = result.dropna()
        return result

    def alignTimeSeries(self, modelI, modelT, run_day, n_days, column_name=ForecastVariable.active.name):
        dates = [datetime.strptime(run_day, "%m/%d/%y") + timedelta(days=x) for x in range(n_days)]
        model_predictions = []
        count = 0
        day0 = dates[0]
        for date in dates:
            t = (date - day0).days
            while modelT[count] <= t:
                count += 1
                if count == len(modelT):
                    print("Last prediction reached - Number of predictions less than required")
                    model_predictions.append(modelI[count - 1])
                    model_predictions_df = pd.DataFrame()
                    model_predictions_df['date'] = [date.strftime("%-m/%-d/%y") for date in dates]
                    model_predictions_df[column_name] = model_predictions
                    return model_predictions_df

            x0 = modelI[count] - (
                    ((modelI[count] - modelI[count - 1]) / (modelT[count] - modelT[count - 1])) * (modelT[count] - t))
            model_predictions.append(x0)
        model_predictions_df = pd.DataFrame()
        model_predictions_df['date'] = [date.strftime("%-m/%-d/%y") for date in dates]
        model_predictions_df[column_name] = model_predictions

        return model_predictions_df
