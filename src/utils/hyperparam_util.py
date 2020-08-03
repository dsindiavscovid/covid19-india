import numpy as np
from hyperopt import fmin, tpe, space_eval, Trials


def hyperparam_tuning(func, search_space, max_evals, algo=tpe.suggest):
    trials = Trials()
    best = fmin(func, search_space, algo=algo, max_evals=max_evals, trials=trials)
    print("Best fit:", space_eval(search_space, best))
    trial_loss = np.asarray(trials.losses(), dtype=float)
    best_ind = np.argmin(trial_loss)
    best_loss = trial_loss[best_ind]
    print("Best Loss:", best_loss)

    trial_list = []
    for trial in trials:
        temp_params = dict()
        for key in trial['misc']['vals']:
            temp_params[key] = trial['misc']['vals'][key][0]

        trial_list.append((temp_params, trial['result']['loss']))
    trial_list = sorted(trial_list, key=lambda x: x[1])

    result = {"best_params": space_eval(search_space, best),
              "trials": trial_list,
              "best_loss": best_loss
              }
    return result


def hyperparam_tuning_ensemble(func, search_space, max_evals, algo=tpe.suggest):
    trials = Trials()
    best = fmin(func, search_space, algo=algo, max_evals=max_evals, trials=trials)
    trial_list = []
    for trial in trials:
        temp_params = dict()
        for key in trial['misc']['vals']:
            temp_params[key] = trial['misc']['vals'][key][0]

        trial_list.append((temp_params, trial['result']['loss']))
    trial_list = sorted(trial_list, key=lambda x: x[1])

    return trial_list
