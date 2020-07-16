import pandas as pd
from entities.forecast_variables import ForecastVariable


def get_weighted_predictions(predictions_df_dict, weights):
    """Gets weighted predictions for all constituent models

    Args:
        predictions_df_dict (dict): dictionary of the form {index: predictions} for constituent models
        weights (dict): dictionary of the form {index: weight} for constituent models

    Returns:
        dict: dictionary of the form {index: weighted predictions} for constituent models
    """

    for idx in predictions_df_dict:
        predictions_df_dict[idx] = predictions_df_dict[idx].multiply(weights[idx])
    return predictions_df_dict


def create_trials_dataframe(predictions_df_dict, column=ForecastVariable.active):
    """Create dataframe of predictions for a single forecast variable from all constituent models

    Args:
        predictions_df_dict (dict): dictionary of the form {index: predictions} for constituent models
        column (ForecastVariable): predictions for this forecast variable are extracted
            from constituent model predictions

    Returns:
        pd.DataFrame: dataframe of predictions of forecast variable from all constituent models
    """

    trials_df_list = []
    for idx in predictions_df_dict:
        trials_df_list.append(predictions_df_dict[idx].loc[:, [column]].T)
    return pd.concat(trials_df_list, axis=0, ignore_index=True)


def uncertainty_dict_to_df(percentiles_predictions):
    """Convert a dictionary of uncertainty predictions to a dataframe

    Args:
        percentiles_predictions (dict): dictionary of the form {percentile: predictions dataframe}

    Returns:
        pd.DataFrame: dataframe with predictions for percentiles and confidence intervals
    """

    columns = ['predictionDate']
    for decile in percentiles_predictions.keys():
        df_prediction = percentiles_predictions[decile]['df_prediction']
        cols = list(df_prediction.columns)
        for col in cols:
            columns.append(col+'_{}'.format(decile))

    df_output = pd.DataFrame(columns=columns)

    date_series = percentiles_predictions[list(percentiles_predictions.keys())[0]]['df_prediction'].index
    df_output['predictionDate'] = date_series
    df_output.set_index('predictionDate', inplace=True)

    for decile in percentiles_predictions.keys():
        df_prediction = percentiles_predictions[decile]['df_prediction']
        cols = list(df_prediction.columns)
        for col in cols:
            df_output.loc[:, col+'_{}'.format(decile)] = df_prediction[col]
#         df_output.loc[:, f'active_{decile}'] = df_prediction['active']
#         df_output.loc[:, f'hospitalized_{decile}'] = df_prediction['hospitalized']
#         df_output.loc[:, f'icu_{decile}'] = df_prediction['icu']
#         df_output.loc[:, f'recovered_{decile}'] = df_prediction['recovered']
#         df_output.loc[:, f'deceased_{decile}'] = df_prediction['deceased']
#         df_output.loc[:, f'confirmed_{decile}'] = df_prediction['confirmed']

    df_output.reset_index(inplace=True)

    return df_output
