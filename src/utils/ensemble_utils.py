import pandas as pd
from entities.forecast_variables import ForecastVariable


def get_weighted_predictions(predictions_df_dict, weights):
    for idx in predictions_df_dict:
        predictions_df_dict[idx] = predictions_df_dict[idx].multiply(weights[idx])
    return predictions_df_dict


def create_trials_dataframe(predictions_df_dict, column=ForecastVariable.active):
    trials_df_list = []
    for idx in predictions_df_dict:
        trials_df_list.append(predictions_df_dict[idx].loc[:, [column]].T)
    return pd.concat(trials_df_list, axis=0, ignore_index=True)


def uncertainty_dict_to_df(self, percentiles_forecast, icu_fraction=0.02):
    columns = ['Region Type', 'Region', 'Country', 'Lat', 'Long', 'predictionDate']

    for decile in percentiles_forecast.keys():
        columns += [f'active_{decile}',
                    f'hospitalized_{decile}',
                    f'icu_{decile}',
                    f'recovered_{decile}',
                    f'deceased_{decile}',
                    f'total_{decile}',
                    ]
    df_output = pd.DataFrame(columns=columns)

    date_series = percentiles_forecast[list(percentiles_forecast.keys())[0]]['df_prediction'].index
    df_output['predictionDate'] = date_series
    df_output.set_index('predictionDate', inplace=True)

    for decile in percentiles_forecast.keys():
        df_prediction = percentiles_forecast[decile]['df_prediction']
        df_output.loc[:, f'active_{decile}'] = df_prediction['hospitalized']
        df_output.loc[:, f'hospitalized_{decile}'] = df_prediction['hospitalized']
        df_output.loc[:, f'icu_{decile}'] = icu_fraction * df_prediction['hospitalized']
        df_output.loc[:, f'recovered_{decile}'] = df_prediction['recovered']
        df_output.loc[:, f'deceased_{decile}'] = df_prediction['deceased']
        df_output.loc[:, f'total_{decile}'] = df_prediction['confirmed']

    df_output.reset_index(inplace=True)

    return df_output
