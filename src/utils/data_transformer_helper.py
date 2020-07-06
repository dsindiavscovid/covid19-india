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
    preddf.rename_axis(columns={"date": None}, inplace=True) # if no json conversion
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
    preddf.rename_axis(columns={"date": None}, inplace=True) # if no json conversion
    return preddf


def convert_dataframe( region_df):
    region_df = region_df.reset_index()
    region_df.drop(["region_name", "region_type", "index"], axis=1, inplace=True)
    headers = region_df.transpose().reset_index().iloc[0]
    transposed_df = pd.DataFrame(region_df.transpose().reset_index().values[1:], columns=headers)
    transposed_df.rename({"observation": "date"}, axis='columns', inplace=True)
    return transposed_df