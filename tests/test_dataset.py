import context
from tqdm import tqdm
from pynuscenes.nuscenes_dataset import NuscenesDataset
from pynuscenes.utils.visualize import show_sample_data

def test_dataset():
    dataset_location = '../../data/nuscenes'
    
    mini_dataset_vehicle = NuscenesDataset(cfg='../pynuscenes/config/cfg.yml')
    
    for sample in tqdm(mini_dataset_vehicle):
        show_sample_data(sample, coordinates='vehicle')
        input('press enter to continue')
##------------------------------------------------------------------------------
if __name__ == "__main__":
    test_dataset()