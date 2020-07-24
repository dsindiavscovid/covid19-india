import collections

import pandas as pd

from entities.forecast_variables import ForecastVariable


def convert_to_initial_observations(df):
    df = df.transpose().reset_index()
    headers = df.iloc[0]
    headers[0] = "observation"
    new_df = pd.DataFrame(df.values[1:], columns=headers)
    return new_df


def convert_to_jhu_format_with_min_max(predictions_df, region_type, region_name, mape):
    dates = predictions_df['date']
    preddf = predictions_df.set_index('date')
    columns = [ForecastVariable.active.name, ForecastVariable.hospitalized.name, ForecastVariable.icu.name,
               ForecastVariable.recovered.name, ForecastVariable.deceased.name, ForecastVariable.confirmed.name]
    for col in columns:
        preddf = preddf.rename(columns={col: col + '_mean'})
    preddf = preddf.transpose().reset_index()
    preddf = preddf.rename(columns={"index": "prediction_type", })
    error = min(1, float(mape) / 100)
    for col in columns:
        col_mean = col + '_mean'
        series = preddf[preddf['prediction_type'] == col_mean][dates]
        newSeries = series.multiply((1 - error))
        newSeries['prediction_type'] = col + '_min'
        preddf = preddf.append(newSeries, ignore_index=True)
        newSeries = series.multiply((1 + error))
        newSeries['prediction_type'] = col + '_max'
        preddf = preddf.append(newSeries, ignore_index=True)
        preddf = preddf.rename(columns={col: col + '_mean'})
    preddf.insert(0, 'Region Type', region_type)
    preddf.insert(1, 'Region', " ".join(region_name))
    preddf.insert(2, 'Country', 'India')
    preddf.insert(3, 'Lat', 20)
    preddf.insert(4, 'Long', 70)
    preddf.rename_axis(columns={"date": None}, inplace=True)  # if no json conversion
    return preddf


def convert_to_required_format(predictions_df, region_type, region_name):
    preddf = predictions_df.set_index('date')
    columns = [ForecastVariable.active.name, ForecastVariable.hospitalized.name, ForecastVariable.icu.name,
               ForecastVariable.recovered.name, ForecastVariable.deceased.name, ForecastVariable.confirmed.name]
    for col in columns:
        preddf = preddf.rename(columns={col: col + '_mean'})
    preddf.insert(0, 'Region Type', region_type)
    preddf.insert(1, 'Region', " ".join(region_name))
    preddf.insert(2, 'Country', 'India')
    preddf.insert(3, 'Lat', 20)
    preddf.insert(4, 'Long', 70)
    return preddf


def convert_to_jhu_format(predictions_df, region_type, region_name):
    preddf = predictions_df.set_index('date')
    columns = [ForecastVariable.active.name, ForecastVariable.hospitalized.name, ForecastVariable.icu.name,
               ForecastVariable.recovered.name, ForecastVariable.deceased.name, ForecastVariable.confirmed.name]
    for col in columns:
        preddf = preddf.rename(columns={col: col + '_mean'})
    preddf = preddf.transpose().reset_index()
    preddf = preddf.rename(columns={"index": "prediction_type", })
    preddf.insert(0, 'Region Type', region_type)
    preddf.insert(1, 'Region', " ".join(region_name))
    preddf.insert(2, 'Country', 'India')
    preddf.insert(3, 'Lat', 20)
    preddf.insert(4, 'Long', 70)
    preddf.rename_axis(columns={"date": None}, inplace=True)  # if no json conversion
    return preddf


def convert_to_old_required_format(self, run_day, predictions_df, region_type, region_name):

    preddf = predictions_df.set_index('date')
    columns = [ForecastVariable.active.name, ForecastVariable.hospitalized.name,
               ForecastVariable.recovered.name, ForecastVariable.deceased.name, ForecastVariable.confirmed.name]
    for col in columns:
        preddf = preddf.rename(columns={col: col + '_mean'})
    error = min(1, float(self._model_parameters['MAPE']) / 100)
    for col in columns:
        col_mean = col + '_mean'
        preddf[col + '_min'] = preddf[col_mean] * (1 - error)
        preddf[col + '_max'] = preddf[col_mean] * (1 + error)

    preddf.insert(0, 'run_day', run_day)
    preddf.insert(1, 'Region Type', region_type)
    preddf.insert(2, 'Region', " ".join(region_name))
    preddf.insert(3, 'Model', self._model.__class__.__name__)
    preddf.insert(4, 'Error', "MAPE")
    preddf.insert(5, "Error Value", error * 100)

    return preddf


def convert_dataframe(region_df):
    region_df = region_df.reset_index()
    region_df.drop(["region_name", "region_type", "index"], axis=1, inplace=True)
    headers = region_df.transpose().reset_index().iloc[0]
    transposed_df = pd.DataFrame(region_df.transpose().reset_index().values[1:], columns=headers)
    transposed_df.rename({"observation": "date"}, axis='columns', inplace=True)
    return transposed_df


def loss_json_to_dataframe(loss_dict, name):
    loss_df = pd.DataFrame(index=[name])
    for loss in loss_dict:
        if len(loss['variable_weights']) > 1:
            continue
        loss_df[loss['variable_weights'][0]['variable'].name] = loss['value']
    return loss_df


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_train_loss_config(config):
    items = [('metric_name', config['metric_name'])]
    for variable in config['variable_weights']:
        items.append((variable['variable']+'_weight', variable['weight']))
    return dict(items)


def flatten_eval_loss_config(config):
    items = []
    for loss in config:
        variable = loss['variable_weights'][0]
        items.append((variable['variable']+'_metric_name', loss['metric_name']))
        items.append((variable['variable']+'_weight', variable['weight']))
    return dict(items)

