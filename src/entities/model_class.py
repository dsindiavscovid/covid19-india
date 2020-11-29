import enum


@enum.unique
class ModelClass(str, enum.Enum):
    SEIR = "SEIR"
    IHME = "IHME"
    SEIHRD = "SEIHRD"
    heterogeneous_ensemble = "heterogeneous_ensemble"
    homogeneous_ensemble = "homogeneous_ensemble"
    SEIHRD_gen = "SEIHRD_gen"
