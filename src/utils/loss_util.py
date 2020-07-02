from typing import List

from entities.loss_function import LossFunction
from utils import metrics_util
from utils.data_transformer_helper import convert_dataframe


def evaluate_for_forecast(observations, predictions_df, loss_functions: List[LossFunction]):
    metrics_results = []
    actual_df = convert_dataframe(observations)
    actual_df = actual_df[actual_df.date.isin(predictions_df.date)]
    if actual_df.shape[0] != predictions_df.shape[0]:
        raise Exception("Error in evaluation: number of rows don't match in predictions and actual dataframe")
    for loss_function in loss_functions:
        value = 0
        for variable_weight in loss_function.variable_weights:
            variable_name = variable_weight.variable.name
            value += variable_weight.weight * (
                getattr(metrics_util, loss_function.metric_name.name)(actual_df[variable_name].values,
                                                                      predictions_df[variable_name].values))
        loss_function.value = value
        metrics_results.append(loss_function.dict())
    return metrics_results
