import yaml
from easydict import EasyDict as edict

def yaml_load(fileName, safe_load=False):
    """
    Load a yaml file as an EasyDict object with dot-notation access.
    
    :param fileName (str): yaml filename
    :param safe_load (bool): Use yaml safe_load
    """
    fc = None
    with open(fileName, 'r') as f:
        if safe_load:
            fc = edict(yaml.safe_load(f))
        else:
            fc = edict(yaml.load(f))

    return fc