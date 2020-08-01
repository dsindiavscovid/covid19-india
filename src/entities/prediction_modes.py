import enum


@enum.unique
class PredictionMode(str, enum.Enum):
    predictions_with_uncertainty = "predictions_with_uncertainty"
    best_fit = "best_fit"
    mean_predictions = "mean_predictions"
    params_with_uncertainty = "params_with_uncertainty"
    mean_params = "mean_params"
