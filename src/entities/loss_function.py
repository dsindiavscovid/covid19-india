from typing import Dict

from pydantic import BaseModel

from entities.forecast_variables import ForecastVariable
from entities.metric_name import MetricName


class VariableWeight(BaseModel):
    variable: ForecastVariable
    weight: float


class LossFunction(BaseModel):
    metric_name: MetricName
    weights: Dict[ForecastVariable, int]
    value: float = 0.0
