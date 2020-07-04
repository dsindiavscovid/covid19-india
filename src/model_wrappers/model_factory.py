from entities.model_class import ModelClass
from model_wrappers.intervention_enabled_seihrd import InterventionEnabledSEIHRD
from model_wrappers.intervention_enabled_seir import InterventionEnabledSEIR
from model_wrappers.seir import SEIR
from model_wrappers.seihrd import SEIHRD
from model_wrappers.heterogeneous_ensemble import HeterogeneousEnsemble
from model_wrappers.homogeneous_ensemble import HomogeneousEnsemble


class ModelFactory:

    @staticmethod
    def get_model(model_class: ModelClass, model_parameters):
        if model_class.__eq__(ModelClass.SEIR):
            return SEIR(model_parameters)
        elif model_class.__eq__(ModelClass.SEIHRD):
            return SEIHRD(model_parameters)
        elif model_class.__eq__(ModelClass.heterogeneous_ensemble):
            return HeterogeneousEnsemble(model_parameters)
        elif model_class.__eq__(ModelClass.homogeneous_ensemble):
            return HomogeneousEnsemble(model_parameters)
        else:
            raise Exception("Model Class is not in supported classes {}".format(["SEIR"]))

    @staticmethod
    def get_intervention_enabled_model(model_class: ModelClass, model_parameters):
        if model_class.__eq__(ModelClass.SEIR):
            return InterventionEnabledSEIR(model_parameters)
        if model_class.__eq__(ModelClass.SEIHRD):
            return InterventionEnabledSEIHRD(model_parameters)
        else:
            raise Exception("Model Class is not in supported classes {}".format(["SEIR"]))
