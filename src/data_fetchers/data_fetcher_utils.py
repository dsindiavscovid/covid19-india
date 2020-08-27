import json
import urllib.request


def get_raw_data_dict(input_url):
    """Gets dictionary of raw data from a given URL"""
    with urllib.request.urlopen(input_url) as url:
        data_dict = json.loads(url.read().decode())
        return data_dict


def load_regional_metadata(file_path):
    """Gets dictionary of regional metadata from a given file path"""
    with open(file_path, 'r') as fp:
        return json.load(fp)
