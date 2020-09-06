import context
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynuscenes.utils.visualize as nsvis
from pynuscenes.utils.io_utils import save_fig
import pynuscenes.utils.nuscenes_utils as nsutils
from pynuscenes.nuscenes_dataset import NuscenesDataset

##------------------------------------------------------------------------------
def test_visualization(nusc):

    for sample in tqdm(nusc):
        ## Render the whole sample using nuscenes devkit API
        sample_token = sample['sample_token']
        nusc.render_sample(sample_token)
        # plt.show(block=False)
        plt.savefig('0_sample.jpg')

        ## Render point cloud on image using nuscenes devkit API
        nusc.render_pointcloud_in_image(sample['sample_token'],
                                        pointsensor_channel = 'RADAR_FRONT',
                                        camera_channel = 'CAM_FRONT',
                                        dot_size = 8)
        # plt.show(block=False)
        plt.savefig('0_camera_radar.jpg')
        plt.close()

        ## Render one sensor using nuscenes devkit API
        sample_data_token = sample['camera'][2]['token']
        nusc.render_sample_data(sample_data_token)
        # plt.show(block=False)
        plt.savefig('0_camera.jpg')
        plt.close()

        ## Render sample using nuscenes_dataset API in 3D
        nsvis.render_sample_in_3d(sample, 
                            coordinates=nusc.cfg.COORDINATES)
        
        # plt.show(block=False)
        input('press enter to continue')
        plt.close()
##------------------------------------------------------------------------------
def test_new_viz(nusc):
    
    for i in range(10, len(nusc)):
        sample = nusc[i]
        print('sample {}'.format(i))
        ## Render whole sample is 2D
        figure = nsvis.render_sample_in_2d(sample, out_path='1_camera_radar.jpg')
        
        plt.cla()
        input('here')

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
def test_database(nusc):
    frames = nusc.db['frames']
    meta = nusc.db['metadata']
    print('Number of frames:', len(frames))
    print('Metadata:')
    for key,val in meta.items():
        print('    ', key,': ',val)

    for i, frame in enumerate(frames):
        print('frame {}:'.format(i) )
        for key, val in frame.items():
            value = len(val) if isinstance(val, list) else val
            print('    ', key, ': {}'.format(value))
        # print(frame)
        input('here')
##------------------------------------------------------------------------------
def test_dataset_mapper(nusc):
    import time
    
    ## Test average sample loading time
    # total=0.0
    # for i in range(len(nusc)):
    #     start = time.time()
    #     sample = nusc[i]
    #     end = time.time()
    #     total += end - start
    # print('Average time: ', total/i)

    ## Test data mapping correctness by visualization
    for i in range(len(nusc)):
        sample = nusc[i]
        print(len(sample['anns']))
        
        figure = nsvis.render_sample_in_2d(sample, out_path='1_camera_radar.jpg')
        
        ## Render point cloud on image using nuscenes devkit API
        # nusc.render_pointcloud_in_image(sample['sample_token'],
        #                                 pointsensor_channel = 'LIDAR_TOP',
        #                                 camera_channel = sample['camera'][0]['channel'],
        #                                 dot_size = 1)
        # # plt.show(block=False)
        # plt.savefig('0_camera_lidar.jpg')

        input('here')
##------------------------------------------------------------------------------
if __name__ == "__main__":
    nusc = NuscenesDataset(dataroot='../../datasets/nuscenes_mini',
                           version="v1.0-mini",
                           split='mini_val',
                           cfg='pynuscenes/config/cfg.yml')
    
    test_visualization(nusc)
    # test_points_in_image(nusc)
    # test_database(nusc)
    # test_dataset_mapper(nusc)
    # test_new_viz(nusc)