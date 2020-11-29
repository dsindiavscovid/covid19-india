from typing import Dict

from entities.forecast_variables import ForecastVariable
from entities.metric_name import MetricName
from pydantic import BaseModel


class VariableWeight(BaseModel):
    variable: ForecastVariable
    weight: float


class LossFunction(BaseModel):
    metric_name: MetricName
    weights: Dict[ForecastVariable, float]
    value: float = 0.0
