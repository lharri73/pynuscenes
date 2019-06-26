################################################################################
## Date Created  : Sat Jun 14 2019                                            ##
## Authors       : Landon Harris, Ramin Nabati                                ##
## Last Modified : Sat Jun 26 2019                                            ##
## Copyright (c) 2019                                                         ##
################################################################################

import io
import logging
import os
import pickle

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud, RadarPointCloud
from PIL import Image
from pyquaternion import Quaternion

from .nuscenes_db import NuscenesDB
from .utils import constants as _C
from .utils import init_logger

class NuscenesDataset(NuscenesDB):
    def __init__(self, 
                 nusc_path, 
                 nusc_version='v1.0-mini', 
                 split='mini_train',
                 db_file=None, 
                 coordinates='vehicle',
                 nsweeps_lidar=1,
                 nsweeps_radar=1, 
                 sensors_to_return=['lidar','radar','camera']) -> None:
        """
        Nuscenes Dataset object to get tokens for every sample in the nuscenes dataset
        :param nusc_path: path to the nuscenes rooth
        :param nusc_version: nuscenes dataset version
        :param split: split in the dataset to use
        :param db_file: location of the db pkl file (will create if none)
        :param coordinates: coordinate system to return all data in
        :param nsweeps_lidar: number of sweeps to use for the LDIAR
        :param nsweeps_radar: number of sweeps to use for the Radar
        :param sensors_to_return: a list of sensor modalities to return (will skip all others)
        """

        self.available_coordinates = ['vehicle', 'global']
        assert coordinates in self.available_coordinates, \
            'Coordinate system not available.'
        assert split in _C.NUSCENES_SPLITS[nusc_version], \
            'Invalid split specified'
        self.sensors_to_return = sensors_to_return
        self.logger = init_logger.initialize_logger('pynuscenes')
        self.coordinates = coordinates
        self.nusc_path = nusc_path
        self.split = split
        self.nusc_version = nusc_version
        self.nsweeps_lidar = nsweeps_lidar
        self.nsweeps_radar = nsweeps_radar
        
        self.radar_min_distance = 1
        self.lidar_min_distance = 1
        
        super().__init__(nusc_path, nusc_version, split)
        
        if db_file is None:
            super().generate_db()
        elif  os.path.exists(db_file):
            with open(db_file, 'rb') as f:
                self.db = pickle.load(f)
        else:
            self.logger.warning('{} does not exist, not saving'.format(db_file))
            super().generate_db()

    def __getitem__(self, idx):
        self.logger.debug('retrieveing id: {}'.format(idx))
        assert idx <= len(self), 'Requested dataset index out of range'
        return self.get_sensor_data(idx)

    def __len__(self):
        return len(self.db['frames'])

    ##--------------------------------------------------------------------------
    def get_sensor_data(self, idx: int) -> dict:
        """
        Returns sensor data in vehicle or global coordinates
        :param idx: id of the dataset's split to retrieve
        :return sensor_data: dictionary containing all sensor data for that frame
        """

        frame = self.db['frames'][idx]
        sensor_data = {
            "lidar": {
                "points": None,
                "sweeps": [],
                "cs_record": None,
            },
            "camera": [{
                "image": None,
                "camera_name": cam,
                "cs_record": None,
                "sweeps": []
            } for cam in _C.CAMERAS.keys()],
            "radar":{
                "points": None,
                'sweeps': []
            },
            "annotations": None,
            "ego_pose": None,
            "id": frame["id"]
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
        if 'lidar' in self.sensors_to_return:
            sensor_data['lidar']['points'], sensor_data['lidar']['cs_record'] = \
                self._get_lidar_data(sample_rec, lidar_sample_data, pose_rec, self.nsweeps_lidar)

        ## Get camera data
        if 'camera' in self.sensors_to_return:
            for i, cam in enumerate(_C.CAMERAS.keys()):
                image, cs_record = self._get_cam_data(frame['sample'][cam])
                sensor_data['camera'][i]['image'] = image
                sensor_data['camera'][i]['cs_record'] = cs_record

        ## Get Radar data
        if 'radar' in self.sensors_to_return:
            sensor_data['radar']['points'] = self._get_all_radar_data(frame,
                                                                    sample_rec,
                                                                    pose_rec,
                                                                    self.nsweeps_radar)
       ## Get annotations
        sensor_data["annotations"] = self._get_annotations(frame, pose_rec)
        # print('nuscenes dataset', res['lidar']['points'].points.shape)
        return sensor_data

    ##--------------------------------------------------------------------------
    def _get_annotations(self, frame: dict, pose_rec: dict) -> [Box]:
        """
        Gets the annotations for this sample in the vehicle coordinates
        :param frame: the frame returned from the db for this sample
        :param pose_record: ego pose record dictionary from nuscenes
        :return: list of Nuscenes Boxes
        """
        if self.split == 'test':
            return []
        else:
            box_list = []
            ## Get boxes from nuscenes in Global coordinates
            orig_box_list = self.nusc.get_boxes(frame['sample']['LIDAR_TOP'])
            for box in orig_box_list:
                ## Filter boxes based on their class
                try:
                    box.name = _C.NAMEMAPPING[box.name]
                except KeyError:
                    continue
                
                box.velocity = self.nusc.box_velocity(box.token)
                
                ## Global to Vehicle
                if self.coordinates == 'vehicle':
                    box.translate(-np.array(pose_rec['translation']))
                    box.rotate(Quaternion(pose_rec['rotation']).inverse)

                box_list.append(box)
                
        return box_list

    ##--------------------------------------------------------------------------
    def _get_all_radar_data(self, frame: dict, sample_rec: str, pose_rec, nsweeps_radar: int) -> RadarPointCloud:
        """
        Concatenates all radar pointclouds from this sample into one pointcloud
        :param frame: the frame returned from the db for this sample
        :param sample_rec: the sample record dictionary from nuscenes
        :param pose_rec: ego pose record dictionary from nuscenes
        :param nsweeps_radar: number of sweeps to retrieve for each radar
        :return: RadarPointCloud with all points
        """

        all_radar_pcs = RadarPointCloud(np.zeros((18, 0)))
        
        for radar in _C.RADARS.keys():
            sample_data = self.nusc.get('sample_data', frame['sample'][radar])
            current_radar_pc = self._get_radar_data(sample_rec, 
                                                    sample_data, 
                                                    nsweeps_radar)
            ## Vehicle to global
            if self.coordinates == 'global':
                current_radar_pc.rotate(Quaternion(pose_rec['rotation']).rotation_matrix)
                current_radar_pc.translate(np.array(pose_rec['translation']))

            all_radar_pcs.points = np.hstack((all_radar_pcs.points, 
                                              current_radar_pc.points))
        
        return all_radar_pcs
    
    ##--------------------------------------------------------------------------
    def _get_radar_data(self, sample_rec: dict, sample_data: dict, nsweeps: int) -> RadarPointCloud:
        """
        Returns Radar point cloud in Vehicle Coordinates
        :param sample_rec: sample record dictionary from nuscenes
        :param sample_data: sample data dictionary from nuscenes
        :param nsweeps: number of sweeps to return for this radar
        :return pc: RadarPointCloud containing this samnple and all sweeps
        """

        radar_path = os.path.join(self.nusc_path, sample_data['filename'])
        cs_record = self.nusc.get('calibrated_sensor', 
                                  sample_data['calibrated_sensor_token'])
        
        if nsweeps > 1:
            ## Returns in vehicle coordinates
            pc, _ = RadarPointCloud.from_file_multisweep(self.nusc,
                                                      sample_rec, 
                                                      sample_data['channel'], 
                                                      sample_data['channel'], 
                                                      nsweeps=nsweeps,
                                                      min_distance=self.radar_min_distance)
        else:
            ## Returns in sensor coordinates
            pc = RadarPointCloud.from_file(radar_path)
            ## Sensor to vehicle
            rot_matrix = Quaternion(cs_record['rotation']).rotation_matrix
            pc.rotate(rot_matrix)
            pc.translate(np.array(cs_record['translation']))


        return pc
        
    ##--------------------------------------------------------------------------
    def _get_lidar_data(self, sample_rec: dict, sample_data: dict, pose_rec: dict, nsweeps:int =1) -> LidarPointCloud:
        """
        Returns the LIDAR pointcloud for this frame in vehicle/global coordniates
        :param sample_rec: sample record dictionary from nuscenes
        :param sample_data: sample data dictionary from nuscenes
        :param pose_rec: ego pose record dictionary from nuscenes
        :param nsweeps: number of sweeps to return for the LIDAR
        :return: LidarPointCloud containing this sample and all sweeps
        """

        lidar_path = os.path.join(self.nusc_path, sample_data['filename'])
        cs_record = self.nusc.get('calibrated_sensor', 
                                  sample_data['calibrated_sensor_token'])
        if nsweeps > 1:
            ## Returns in vehicle
            lidar_pc, _ = LidarPointCloud.from_file_multisweep(self.nusc,
                                                        sample_rec, 
                                                        sample_data['channel'], 
                                                        sample_data['channel'], 
                                                        nsweeps=nsweeps)
        else:
            ## returns in sensor coordinates
            lidar_pc = LidarPointCloud.from_file(lidar_path)
            ## Sensor to vehicle
            lidar_pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
            lidar_pc.translate(np.array(cs_record['translation']))
       
        ## Vehicle to global
        if self.coordinates == 'global':
            lidar_pc.rotate(Quaternion(pose_rec['rotation']).rotation_matrix)
            lidar_pc.translate(np.array(pose_rec['translation']))

        return lidar_pc, cs_record

    ##--------------------------------------------------------------------------
    def _get_cam_data(self, cam_token: str) -> (np.ndarray, np.ndarray):
        """
        :param cam_token: sample data token for this camera
        :return image, intrinsics: numpy array of the image intrinsics
        """

        ## Get camera image
        cam_path = self.nusc.get_sample_data_path(cam_token)
        cam_data = self.nusc.get('sample_data', cam_token)
        cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        if os.path.exists(cam_path):
            with open(cam_path, 'rb') as f:
                image_str = f.read()
        else:
            raise Exception('Camera image not found at {}'.format(cam_path))
        image = np.array(Image.open(io.BytesIO(image_str)))
        return image, cs_record
    
    ##--------------------------------------------------------------------------
    @staticmethod
    def pc_to_sensor(pc, cs_record, global_coordinates=False, ego_pose=None):
        """
        Tramsform the iput point cloud from global/vehicle coordinates to
        sensor coordinates
        """
        assert global_coordinates is False or (global_coordinates and ego_pose is not None), \
            'when in global coordinates, ego_pose is required'
        
        if global_coordinates:
            ## Transform from global to vehicle
            pc.translate(np.array(-np.array(ego_pose['translation'])))
            pc.rotate(Quaternion(ego_pose['rotation']).rotation_matrix.T)

        ## Transform from vehicle to sensor
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        return pc
