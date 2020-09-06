import yaml
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

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
##------------------------------------------------------------------------------
def save_fig(filepath, fig=None, format='pdf'):
    '''
    Save Matplotlib figure with no whitespace in different formats
    '''
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0,0,1,1,0,0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(filepath, pad_inches = 0, bbox_inches='tight', format=format)