import context
from tqdm import tqdm
import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynuscenes.utils.nuscenes_utils as nsutils
from pynuscenes.utils.io_utils import save_fig
from pynuscenes.nuscenes_dataset import NuscenesDataset
from pynuscenes.utils.visualize import visualize_sample_3d, visualize_sample_2d
from pynuscenes.utils.visualize import draw_gt_box_on_image

def test_dataset():    
    mini_dataset_vehicle = NuscenesDataset(dataroot='../data/nuscenes',
                                           cfg='../pynuscenes/config/cfg.yml')
    for sample in tqdm(mini_dataset_vehicle):
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
        
        ## Render nuscenes_dataset sample using nuscenes_dataset API in 3D
        visualize_sample_3d(sample, coordinates='vehicle')
        input('press enter to continue')
        
        ## Render nuscenes_dataset sample using nuscenes_dataset API in 2D
        # figure = visualize_sample_2d(sample, coordinates='vehicle', out_path='output.jpg')
        # plt.show(block=False)
        # input('press enter to continue')
        # plt.close(fig=figure)

##------------------------------------------------------------------------------
if __name__ == "__main__":
    test_dataset()