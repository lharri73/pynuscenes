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

def test_dataset(nusc):    

    for sample in tqdm(nusc):
        ## Render sample using nuscenes devkit API
        # sample_token = sample['sample_token']
        # nusc.render_sample(sample_token)
        # plt.show(block=False)

        sample_data_token = sample['camera'][0]['token']
        nusc.render_sample_data(sample_data_token)
        plt.show(block=False)

        # nusc.render_pointcloud_in_image(sample['sample_token'],
        #                                                 pointsensor_channel = 'RADAR_FRONT',
        #                                                 camera_channel = 'CAM_FRONT',
        #                                                 dot_size = 8)
        # plt.show()
        # input('here')

        ## Render nuscenes_dataset sample using nuscenes_dataset API in 3D
        # visualize_sample_3d(sample, 
        #                     coordinates=nusc.cfg.COORDINATES)
        input('press enter to continue')
        
        ## Render nuscenes_dataset sample using nuscenes_dataset API in 2D
        figure = visualize_sample_2d(sample, 
                                     coordinates=nusc.cfg.COORDINATES, 
                                     out_path='../output/figure.jpg')
        plt.show(block=False)
        input('press enter to continue')
        plt.close(fig=figure)
##------------------------------------------------------------------------------
def test_points_in_image(nusc):
    from pynuscenes.utils.nuscenes_utils import (points_in_image, 
                                                 vehicle_to_sensor, 
                                                 map_pointcloud_to_image)
    for sample in nusc:
        radar_points = sample['radar']['pointcloud']
        radar_pose_record = sample['radar']['pose_record']
        cam_cs_record = sample['camera'][0]['cs_record']
        cam_pose_record = sample['camera'][0]['pose_record']
        points_veh = vehicle_to_sensor(radar_points, cam_cs_record)

        ## Filter points
        mask1 = points_in_image(points_veh, cam_cs_record)
        points = radar_points.points[:, mask1]
        print(mask1)
        input('here')
##------------------------------------------------------------------------------
if __name__ == "__main__":
    nusc = NuscenesDataset(dataroot='../data/nuscenes',
                           cfg='../pynuscenes/config/cfg.yml')
    # test_dataset(nusc)
    test_points_in_image(nusc)