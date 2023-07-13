import yaml


def create_config_from_yaml():
    """Loads data from configuration.yaml"""
    with open('../config/configuration.yaml') as config_file:
        data = yaml.safe_load(config_file)
    return data
