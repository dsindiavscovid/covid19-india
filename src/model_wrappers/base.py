import pandas as pd

import abc

from entities.forecast_variables import ForecastVariable
from entities.loss_function import LossFunction

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class ModelWrapperBase(ABC):
    @property
    @abc.abstractmethod
    def supported_forecast_variables(self):
        pass

    @abc.abstractmethod
    def predict(self, region_metadata: dict, region_observations: pd.DataFrame, run_day: str, start_date: str,
                end_date: str, **kwargs):
        pass

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def is_black_box(self):
        pass

    @abc.abstractmethod
    def train(self, region_metadata: dict, region_observations: pd.DataFrame, train_start_date: str,
              train_end_date: str, search_space: dict, search_parameters: dict, train_loss_function: LossFunction):
        pass
