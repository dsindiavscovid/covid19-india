import enum


@enum.unique
class TrainingMode(str, enum.Enum):
    only_beta = "only_beta"
    full = "full"
    constituent_models = "constituent_models"
