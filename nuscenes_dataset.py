import os
import pickle
import numpy as np
import io
import cv2

from PIL import Image
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from datasets.nuscenes_imdb import NuscenesIMDB
# from utils.visualize import visualize_frame_data
import utils.visualize as vis
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud, Box


class NuscenesDataset(NuscenesIMDB):
    
    NAMEMAPPING = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DETECTION_NAMES = {'car': 1, 
                       'truck': 2, 
                       'bus': 3,
                       'trailer': 4, 
                       'construction_vehicle': 5, 
                       'pedestrian': 6, 
                       'motorcycle': 7, 
                       'bicycle': 8,
                       'traffic_cone': 9,
                       'barrier': 10}
   
    def __init__(self, 
                 root_path, 
                 nusc_version='mini', 
                 split='train',
                 imdb_path=None, 
                 coordinates='vehicle',
                 nsweeps_lidar=1,
                 nsweeps_radar=1):
        
        assert coordinates in ['vehicle', 'global'], 'Coordinates not available.'
        self.coordinates = coordinates
        self.root_path = root_path
        self.split = split
        self.nusc_version = nusc_version
        self.nsweeps_lidar = nsweeps_lidar
        self.nsweeps_radar = nsweeps_radar
        
        self.radar_min_distance = 1
        self.lidar_min_distance = 1
        
        super().__init__(root_path, "v1.0-" + nusc_version)
        
        if imdb_path is None:
            imdb_path = os.path.join(root_path, "%s_imdb_%s.pkl"%(nusc_version, 
                                                                  split))
        
        if not os.path.exists(imdb_path):
            print("IMDB pkl file does not exist; creating...")

            super().generate_imdb()
            self.imdb = self.imdb[split]
            print("IMDB file created.")
        
        else:
            print("IMDB pkl file already exists.")
            with open(imdb_path, 'rb') as f:
                self.imdb = pickle.load(f)


    def __getitem__(self, idx):
        return self.get_sensor_data(idx)


    def __len__(self):
        return len(self.imdb['frames'])

    ##--------------------------------------------------------------------------
    def get_sensor_data(self, idx):
        """
        Returns sensor data in vehicle or global coordinates
        """
        frame = self.imdb['frames'][idx]
        sensor_data = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "sweeps": []
            },
            "camera": [{
                "type": "camera",
                "image": None,
                "camera_name": cam,
                "intrinsics": None,
            } for cam in self.CAMS],
            "radar":{
                "type": "radar",
                "points": None,
                'sweeps': []
            },
            "annotations": None,
            "sweep_annotations": [],
            "ego_pose": None,
            "metadata": {
                "id": frame["id"]
            },
        }
        
        ## Get sample and ego pose data
        lidar_sample_data = self.nusc.get('sample_data', 
                                          frame['sample']['LIDAR_TOP'])
        sample_token = lidar_sample_data['sample_token']
        sample_rec = self.nusc.get('sample', sample_token)
        ego_pose_token = lidar_sample_data['ego_pose_token']
        pose_rec = self.nusc.get('ego_pose', ego_pose_token)
        sensor_data['ego_pose'] = {'translation': pose_rec['translation'], 
                                   'rotation': pose_rec['rotation']}

        ## Get LIDAR data
        sensor_data['lidar']['points'] = self._get_lidar_data(sample_rec,
                                                              lidar_sample_data,
                                                              pose_rec,
                                                              self.nsweeps_lidar)
        ## Get camera data
        for i, cam in enumerate(self.CAMS):
            image, intrinsics = self._get_cam_data(frame['sample'][cam])
            sensor_data['camera'][i]['image'] = image
            sensor_data['camera'][i]['intrinsics'] = intrinsics

        ## Get Radar data
        sensor_data['radar']['points'] = self._get_all_radar_data(frame,
                                                                  sample_rec,
                                                                  pose_rec,
                                                                  self.nsweeps_radar)
       ## Get annotations
        sensor_data["annotations"] = self._get_annotations(frame, pose_rec)
        # print('nuscenes dataset', res['lidar']['points'].points.shape)
        return sensor_data

    ##--------------------------------------------------------------------------
    def _get_annotations(self, frame, pose_rec):

        if not frame['is_test']:
            box_list = []
            orig_box_list = self.nusc.get_boxes(frame['sample']['LIDAR_TOP'])
            for box in orig_box_list:
                ## Filter boxes based on their class
                try:
                    box.name = self.NAMEMAPPING[box.name]
                except KeyError:
                    continue
                
                box.velocity = self.nusc.box_velocity(box.token)
                
                # Global to Vehicle
                if self.coordinates == 'vehicle':
                    box.translate(-np.array(pose_rec['translation']))
                    box.rotate(Quaternion(pose_rec['rotation']).inverse)

                box_list.append(box)
                                        
                
        return box_list

    ##--------------------------------------------------------------------------
    def _get_all_radar_data(self, frame, sample_rec, pose_rec, nsweeps_radar):

        all_radar_pcs = RadarPointCloud(np.zeros((18, 0)))
        
        for radar in self.RADARS:
            sample_data = self.nusc.get('sample_data', frame['sample'][radar])
            current_radar_pc = self._get_radar_data(sample_rec, 
                                                    sample_data, 
                                                    nsweeps_radar)
            ## vehicle to global
            if self.coordinates == 'global':
                current_radar_pc.rotate(Quaternion(pose_rec['rotation']).rotation_matrix)
                current_radar_pc.translate(np.array(pose_rec['translation']))

            all_radar_pcs.points = np.hstack((all_radar_pcs.points, 
                                              current_radar_pc.points))
        
        return all_radar_pcs
    
    ##--------------------------------------------------------------------------
    def _get_radar_data(self, sample_rec, sample_data, nsweeps):
        """
        Returns Radar point cloud in Global Coordinates
        """
        radar_path = os.path.join(self.root_path, sample_data['filename'])
        cs_record = self.nusc.get('calibrated_sensor', 
                                  sample_data['calibrated_sensor_token'])
        
        if nsweeps > 1:
            pc, _ = RadarPointCloud.from_file_multisweep(self.nusc,
                                                      sample_rec, 
                                                      sample_data['channel'], 
                                                      sample_data['channel'], 
                                                      nsweeps=nsweeps,
                                                      min_distance=self.radar_min_distance)
        else:
            pc = RadarPointCloud.from_file(radar_path)

        #sensor to vehicle
        rot_matrix = Quaternion(cs_record['rotation']).rotation_matrix
        pc.rotate(rot_matrix)
        pc.translate(np.array(cs_record['translation']))

        return pc
        
    ##--------------------------------------------------------------------------
    def _get_lidar_data(self, sample_rec, sample_data, pose_rec, nsweeps=1):

        lidar_path = os.path.join(self.root_path, sample_data['filename'])
        cs_record = self.nusc.get('calibrated_sensor', 
                                  sample_data['calibrated_sensor_token'])
        if nsweeps > 1:
            lidar_pc, _ = LidarPointCloud.from_file_multisweep(self.nusc,
                                                        sample_rec, 
                                                        sample_data['channel'], 
                                                        sample_data['channel'], 
                                                        nsweeps=nsweeps)
        else:
            lidar_pc = LidarPointCloud.from_file(lidar_path)

        ## sensor to vehicle
        lidar_pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        lidar_pc.translate(np.array(cs_record['translation']))
        
        ## vehicle to global
        if self.coordinates == 'global':
            lidar_pc.rotate(Quaternion(pose_rec['rotation']).rotation_matrix)
            lidar_pc.translate(np.array(pose_rec['translation']))

        return lidar_pc

    ##--------------------------------------------------------------------------
    def _get_cam_data(self, cam_token):
        ## Get camera image
        cam_path, _, cam_intrinsics = self.nusc.get_sample_data(cam_token)
        if Path(cam_path).exists():
            with open(cam_path, 'rb') as f:
                image_str = f.read()
        else:
            raise Exception('Camera image not found at {}'.format(cam_path))
        image = np.array(Image.open(io.BytesIO(image_str)))
        return image, cam_intrinsics

    ##--------------------------------------------------------------------------
    # def _get_sweeps_data(self, frame, return_global):
    #     box_list = []
    #     lidar_pc_sweeps = []

    #     for lidar_token in frame['sweeps']['LIDAR_TOP']:
    #         pose_rec = self.nusc.get('ego_pose', self.nusc.get('sample_data', lidar_token)['ego_pose_token'])
    #         lidar_path = self.nusc.get_sample_data_path(lidar_token)
    #         lidar_data = self.nusc.get('sample_data', lidar_token)
            
    #         box_list.append(self.nusc.get_boxes(lidar_token))            
    #         lidar_pc = LidarPointCloud.from_file(lidar_path)
    #         cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

    #         #sensor to vehicle
    #         lidar_pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    #         lidar_pc.translate(np.array(cs_record['translation']))
    #         if return_global:
    #             ## vehicle to global
    #             lidar_pc.rotate(Quaternion(pose_rec['rotation']).rotation_matrix)
    #             lidar_pc.translate(np.array(pose_rec['translation']))
    #         lidar_pc_sweeps.append(lidar_pc.points)

    #     lidar_sweeps = lidar_pc_sweeps
    #     sweep_annotations = box_list

    #     #Get Radar sweeps and combine into one point cloud
    #     all_radars = []
    #     for radar in self.RADARS:
    #         current_radar = []
            
    #         for radar_token in frame['sweeps'][radar]:
    #             # pose_rec = self.nusc.get('ego_pose', self.nusc.get('sample_data', radar_token)['ego_pose_token'])
    #             radar_path = self.nusc.get_sample_data_path(radar_token)
    #             radar_data = self.nusc.get('sample_data', radar_token)
    #             current_radar_pc = self._get_radar_data(radar_path, radar_data)
    #             if return_global:
    #                 ## vehicle to global
    #                 current_radar_pc.rotate(Quaternion(pose_rec['rotation']).rotation_matrix)
    #                 current_radar_pc.translate(np.array(pose_rec['translation']))
    #             current_radar.append(current_radar_pc.points)

    #         all_radars.append(current_radar)
        
    #     points = []
    #     for sweep in zip(all_radars[0],all_radars[1],all_radars[2],all_radars[3],all_radars[4]):
    #         current_points = np.zeros((18,0))
    #         for pc in sweep:
    #             current_points = np.hstack((current_points, pc))
    #         points.append(current_points)
        
    #     radar_sweeps = points

    #     return lidar_sweeps, radar_sweeps, sweep_annotations



###############################################################################
if __name__ == "__main__":
    dataset = NuscenesDataset(root_path='../../data/datasets/nuscenes', 
                              nusc_version='mini', 
                              imdb_path=None, 
                              split='train',
                              coordinates='vehicle',
                              nsweeps_lidar=5,
                              nsweeps_radar=2)
    
    sensor_data = dataset[19]
    vis.visualize_frame_data(gt_boxes=sensor_data["annotations"],
                        radar_pc=sensor_data['radar']['points'].points,
                        lidar_pc=sensor_data['lidar']['points'].points,
                        #  image=sensor_data['camera'][2]['image'],
                        cam_coord=False)
    input('here')
    # print(dataset.imdb['frames'][0])