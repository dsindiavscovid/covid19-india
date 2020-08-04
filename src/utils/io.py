import json

import pandas as pd
import simplejson


def read_file(file_path, file_format, data_format, **kwargs):
    output = None
    if file_format == "json" and data_format == "dict":
        with open(file_path, 'r') as f_in:
            output = json.load(f_in)
    elif file_format == "csv" and data_format == "dataframe":
        output = pd.read_csv(file_path, **kwargs)
    else:
        pass
    return output


def write_file(data, file_path, file_format, data_format, **kwargs):
    if file_format == "json" and data_format == "dict":
        with open(file_path, 'w') as f_out:
            json.dump(data, f_out, **kwargs)
    elif file_format == "csv" and data_format == "dataframe":
        if len(kwargs) == 0:
            data.to_csv(file_path, index=False)
        else:
            data.to_csv(file_path, **kwargs)
    else:
        pass
    return


def read_config_file(config_file_path):
    with open(config_file_path) as config_file:
        config_data = simplejson.load(config_file, object_pairs_hook=simplejson.OrderedDict)
    return config_data
