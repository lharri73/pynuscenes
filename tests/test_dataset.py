import context
from tqdm import tqdm
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pynuscenes.utils.nuscenes_utils as nsutils
from pynuscenes.nuscenes_dataset import NuscenesDataset
from pynuscenes.utils.visualize import show_sample_data

def test_dataset():    
    mini_dataset_vehicle = NuscenesDataset(dataroot='../data/nuscenes',
                                           cfg='../pynuscenes/config/cfg.yml')
    for sample in tqdm([mini_dataset_vehicle[2]]):
        ## Render sample using nuscenes devkit API
        # sample_token = sample['sample_token']
        # mini_dataset_vehicle.render_sample(sample_token)
        # plt.show()
        # mini_dataset_vehicle.render_pointcloud_in_image(sample['sample_token'],
        #                                                 pointsensor_channel = 'RADAR_FRONT',
        #                                                 camera_channel = 'CAM_FRONT',
        #                                                 dot_size = 8)
        # plt.show()
        # input('here')
        ## Render nuscenes_dataset sample using nuscenes_dataset API
        show_sample_data(sample, coordinates='vehicle')
        input('press enter to continue')
##------------------------------------------------------------------------------
if __name__ == "__main__":
    test_dataset()