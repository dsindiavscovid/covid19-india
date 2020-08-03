from typing import List

import pandas as pd
from entities.model_class import ModelClass
from pydantic import BaseModel


class ModelBuildingSessionOutputArtifacts(BaseModel):
    """
      Config object to capture output artifacts
    """
    cleaned_case_count_file: str
    plot_case_count: str
    plot_M1_CARD: str
    plot_M1_single_hospitalized: str
    plot_M1_single_confirmed: str
    plot_M1_single_deceased: str
    plot_M1_single_recovered: str
    M1_model_params: str
    M1_model: str
    M1_beta_trials: str
    M1_param_ranges: str
    M1_train_config: str
    M1_test_config: str
    M2_full_output_forecast_file: str
    plot_M2_CARD: str
    plot_M2_single_hospitalized: str
    plot_M2_single_confirmed: str
    plot_M2_single_deceased: str
    plot_M2_single_recovered: str
    plot_M2_forecast_CARD: str
    plot_M2_forecast_single_hospitalized: str
    plot_M2_forecast_single_confirmed: str
    plot_M2_forecast_single_deceased: str
    plot_M2_forecast_single_recovered: str
    M2_model_params: str
    M2_model: str
    M2_beta_trials: str
    M2_param_ranges: str
    M2_percentile_params: str
    M2_train_config: str
    M2_forecast_config: str
    plot_planning_pdf_cdf: str
    plot_M2_planning_CARD: str
    plot_M2_scenario_1_CARD: str
    plot_M2_scenario_2_CARD: str
    plot_M2_scenario_3_CARD: str
    M2_planning_model_params: str
    M2_planning_output_forecast_file: str
    staffing_planning: str
    staffing_scenario_1: str
    staffing_scenario_2: str
    staffing_scenario_3: str
    session_log: str
    model_building_report: str
    planning_report: str


class ModelBuildingSessionParams(BaseModel):
    """
      Config object to capture user params for session
    """
    # TODO: elements of dict fields could be additionally validated by creating config class inheriting from BaseModel
    session_name: str = None
    user_name: str
    experiment_name: str
    output_dir: str
    input_file_path: str = None
    interval_to_consider: int

    region_name: List[str]
    region_type: str
    data_source: str

    time_interval_config: dict
    model_class: ModelClass
    model_parameters: dict
    # TODO: choosing to use dict instead of LossFunction for uniform handling
    # TODO: should we move below three items into one unit because they all pertain to training -
    train_loss_function: dict
    search_space: dict
    search_parameters: dict
    eval_loss_functions: List[dict]
    uncertainty_parameters: dict
    planning: dict
    staffing: dict
    publish_flag: bool = True
    comments: str = ''
    input_artifacts: List


class ModelBuildingSessionMetrics(BaseModel):
    """
      Config object to capture session metrics
    """
    M1_beta: float
    M1_losses: pd.DataFrame = None
    M2_beta: float
    M2_losses: pd.DataFrame = None
    M2_scenarios_r0: List = None

    class Config:
        arbitrary_types_allowed = True
