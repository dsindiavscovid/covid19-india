from typing import List

from entities.data_source import DataSource
from entities.intervention_variable import InputType
from entities.loss_function import LossFunction
from entities.model_class import ModelClass
from pydantic import BaseModel


class BaseConfig(BaseModel):
    data_source: DataSource = DataSource.tracker_data_all
    region_name: List[str]
    region_type: str
    model_class: ModelClass
    model_parameters: dict
    output_file_prefix: str = None
    output_dir: str = None
    input_filepath: str = None


class TrainingModuleConfig(BaseConfig):
    train_start_date: str
    train_end_date: str
    search_space: dict
    search_parameters: dict
    train_loss_function: LossFunction
    eval_loss_functions: List[LossFunction]


class ModelEvaluatorConfig(BaseConfig):
    test_run_day: str
    test_start_date: str
    test_end_date: str
    eval_loss_functions: List[LossFunction]


class ForecastingModuleConfig(BaseConfig):
    forecast_run_day: str
    forecast_start_date: str
    forecast_end_date: str
    # forecast_variables: List[ForecastVariable] TODO: Is this necessary


class Intervention(BaseModel):
    intervention_variable: str
    value: float


class ForecastTimeInterval(BaseModel):
    end_date: str
    interventions: List[Intervention]

    def get_interventions_map(self):
        return dict([tuple(intervention.dict().values()) for intervention in self.interventions])


class ScenarioForecastingModuleConfig(BaseConfig):
    forecast_run_day: str
    start_date: str
    time_intervals: List[ForecastTimeInterval]
    input_type: InputType
