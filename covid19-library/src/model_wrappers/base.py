import pandas as pd

import abc

from entities.forecast_variables import ForecastVariable

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class ModelWrapperBase(ABC):
    @property
    @abc.abstractmethod
    def supported_forecast_variables(self):
        pass

    @abc.abstractmethod
    def predict(self, confirmed_data: pd.Series, recovered_data: pd.Series, run_day, start_date, end_date, **kwargs):
        pass

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def is_black_box(self):
        pass
