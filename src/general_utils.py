import os
from datetime import datetime

from IPython.display import Markdown, display
from utils.time import get_date


def create_output_folder(dir_name, path_prefix):
    # Check if the directory exists
    # if yes, raise a warning; else create a new directory
    new_dir = os.path.join(path_prefix, dir_name)
    if os.path.isdir(new_dir):
        msg = (
            f'{new_dir} already exists. Your content might get overwritten.'
            f'Please change the directory name to retain previous content'
        )
        print(msg)
        pass
    else:
        os.mkdir(new_dir)
    return


def get_dict_field(d, key_list):
    """
        Gets a multi-level nested element specified by a list of keys in a dict object to the specified value
    """
    # TODO: Check if this works and also move it to an appropriate utils file
    # There should be alternate robust implementation via JSONPath or something else
    # dicts are mutable objects
    tmp = d
    for i, k in key_list:
        if i == len(key_list) - 1:
            return tmp[k]
        else:
            if k in tmp:
                tmp = tmp[k]
            else:
                return None
    return


def set_dict_field(d, key_list, value):
    """
        Sets a multi-level nested element specified by a list of keys in a dict object to the specified value
    """
    # TODO: Check if this works and also move it to an appropriate utils file
    # There should be alternate robust implementation via JSONPath or something else
    # dicts are mutable objects
    tmp = d
    for i, k in enumerate(key_list):
        if i == len(key_list) - 1:
            tmp[k] = value
        else:
            if k in tmp:
                tmp = tmp[k]
            else:
                tmp[k] = {}
                tmp = tmp[k]

    return d


def render_artifact(artifact_path, artifact_type):
    if artifact_type == '.md':
        with open(artifact_path) as report_file:
            content = report_file.read()
        display((Markdown(content)))
    else:
        # TODO: add rendering code for other artifact_types
        print('Unable to render currently')
        pass
    return


def merge_dicts(dictlist):
    dunion = {}
    for d in dictlist:
        dunion.update(d)
    return dunion


# TODO: flatten -- dicts and dataframes to flat dict
def flatten(d):
    d_flat = {}
    # val - type -dict - continue DFS
    # val - type -data frame - create row/column keys
    # val - type -anything else - terminate and convert value to str
    return d_flat


def compute_dates(time_interval_config):
    """
          Compute all the dates in time_interval_config in a consistent way
    """

    # Note: expectation is that either the user specifies ALL the dates in "direct" mode
    # or we would populate it based on the offsets and reference day
    is_direct_mode = True
    for k in time_interval_config["direct"]:
        if not time_interval_config["direct"][k]:
            is_direct_mode = False
            break

    # if not direct/expert mode, reference_day is set to today if not specified by user
    # and other dates in the direct mode are computed accordingly
    if not is_direct_mode:

        if not time_interval_config["offset_based"]["reference_day"]:
            d0 = datetime.today().strftime("%-m/%-d/%y")
            time_interval_config["offset_based"]["reference_day"] = d0
        else:  # Added
            d0 = time_interval_config["offset_based"]["reference_day"]
        # compute the direct_mode dates based on this
        d = {}
        offsets = time_interval_config["offset_based"]
        d["forecast_end_date"] = get_date(d0, offsets["forecast_period"])
        d["forecast_run_day"] = get_date(d0, 0)
        d["forecast_start_date"] = get_date(d0, 1)
        d["test_end_date"] = get_date(d0, 0)
        d["test_run_day"] = get_date(d0, -offsets["test_period"] - 1)
        d["test_start_date"] = get_date(d0, -offsets["test_period"])
        d["train1_end_date"] = get_date(d0, -offsets["test_period"] - 1)
        d["train1_run_day"] = get_date(d0, -offsets["test_period"] - offsets["train_period"] - 2)
        d["train1_start_date"] = get_date(d0, -offsets["test_period"] - offsets["train_period"] - 1)
        d["train2_end_date"] = get_date(d0, 0)
        d["train2_run_day"] = get_date(d0, -offsets["train_period"] - 1)
        d["train2_start_date"] = get_date(d0, -offsets["train_period"])
        d["plot_start_date_m1"] = get_date(d0, -offsets["test_period"] - offsets["train_period"] - offsets[
            "plot_buffer"] - 1)
        d["plot_start_date_m2"] = get_date(d0, -offsets["train_period"] - offsets["plot_buffer"])
        time_interval_config["direct"] = d

    return time_interval_config
