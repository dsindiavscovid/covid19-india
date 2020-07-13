import enum


@enum.unique
class ModelClass(str, enum.Enum):
    SEIR = "SEIR"
    IHME = "IHME"
    SEIHRD = "SEIHRD"
    heterogeneous_ensemble = "heterogeneous_ensemble"
    homogeneous_ensemble = "homogeneous_ensemble"
    SEIR_gen = "SEIR_gen"
