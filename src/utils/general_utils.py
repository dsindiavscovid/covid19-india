import os

from IPython.display import Markdown, display


def create_output_folder(dir_name, path_prefix):
    # Check if the directory exists
    # if yes, raise a warning; else create a new directory
    new_dir = os.path.join(path_prefix, dir_name)
    if os.path.isdir(new_dir):
        msg = (
            f'{new_dir} already exists. Your content might get overwritten.\n'
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
    # TODO: Check if this works and also move it to an appropriate utils file - check for pydantic
    # There should be alternate robust implementation via JSONPath or something else
    # dicts are mutable objects
    tmp = d
    for i, k in enumerate(key_list):
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
