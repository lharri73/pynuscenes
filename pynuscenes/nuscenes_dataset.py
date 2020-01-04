import io
import os
import cv2
import time
import copy
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm as tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, box_in_image
from nuscenes.utils.data_classes import Box, LidarPointCloud, RadarPointCloud

import pynuscenes.utils.nuscenes_utils as nsutils
from pyquaternion import Quaternion
from pynuscenes.utils import constants as _C
from pynuscenes.utils import log, io_utils



class NuscenesDataset(NuScenes):
    """
    An improved database and dataloader class for nuScenes.
    """
    def __init__(self, dataroot, cfg):
        """
        :param cfg (str): path to the config file
        """
        self.dataroot = dataroot
        self.cfg = io_utils.yaml_load(cfg, safe_load=True)
        self.logger = log.getLogger(__name__)
        self.logger.info('Loading NuScenes')
        self.frame_id = 0
        self.img_id = 0

        assert self.cfg.COORDINATES in ['vehicle', 'global'], \
            'Coordinate system not valid.'
        assert self.cfg.SPLIT in _C.NUSCENES_SPLITS[self.cfg.VERSION], \
            'NuScenes split not valid.'

        super().__init__(version = self.cfg.VERSION,
                         dataroot = self.dataroot,
                         verbose = self.cfg.VERBOSE)
        self.db = self.generate_db()
    ##--------------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Get one sample from the dataset
        
        :param idx (int): index of the sample to return
        :return sample (dict): one sample from the dataset
        """
        assert idx < len(self), 'Requested dataset index out of range'
        return self._get_sensor_data(idx)
    ##--------------------------------------------------------------------------
    def __len__(self):
        """
        Get the number of samples in the dataset
        
        :return len (int): number of sample in the dataset
        """
        return len(self.db['frames'])
    ##--------------------------------------------------------------------------
    def generate_db(self, out_dir=None):
        """
        Read and preprocess the dataset samples and annotations, representing 
        the dataset items in a lightweight, canonical format. This function does 
        not read the sensor data files (e.g., images are not loaded into memory).

        :param out_dir (str): Directory to save the database pickle file
        :return db (dict): a dictionary containing dataset meta-data 
            and a list of dicts, one for each sample in the dataset
        """
        startTime = time.time()
        scenes_list = nsutils.split_scenes(self.scene, self.cfg.SPLIT)
        self.logger.info('Scenes in {} split: {}'.format(self.cfg.SPLIT, 
                                                        str(len(scenes_list))))
        self.logger.info('Creating database')

        ## Loop over all the scenes
        all_frames = []
        for scene in tqdm(scenes_list):
            scene_frames = []
            scene_record= self.get('scene', scene)
            sample_record= self.get('sample', scene_record['first_sample_token'])
            
            ## Loop over all samples in this scene
            has_more_samples = True
            while has_more_samples:
                scene_frames += self._get_sample_frames(sample_record)
                if sample_record['next'] == "":
                    has_more_samples = False
                else:
                    sample_record= self.get('sample', sample_record['next'])
            all_frames += scene_frames

        metadata = {"version": self.cfg.VERSION}
        db = {"frames": all_frames,
              "metadata": metadata}
        
        self.logger.info('Created database in %.1f seconds' % (time.time()-startTime))
        self.logger.info('Number of samples in {} split: {}'.format(self.cfg.SPLIT,
                                                          str(len(all_frames))))
        ## if an output directory is specified, write to a pkl file
        if out_dir is not None:
            self.logger.info('Saving db to pickle file')
            os.mkdirs(out_dir, exist_ok=True)
            db_filename = "{}_db.pkl".format(self.cfg.SPLIT)
            with open(os.path.join(out_dir, db_filename), 'wb') as f:
                pickle.dump(db['test'], f)
        return db
    ##--------------------------------------------------------------------------
    def _get_sample_frames(self, sample_record):
        """
        Get all the frames from a single sample from the nuscenes dataset
        
        :param sample_record (dict): sample record dictionary from nuscenes
        :return frames (list): list of frame dictionaries
        """
        frames = []
        frame = {
            'camera':[],
            'lidar': [],
            'radar':[],
            'coordinates': self.cfg.COORDINATES,
        }
        ## Generate a frame containing all sensors
        frame['anns'] = sample_record['anns']
        frame['sample_token'] = sample_record['token']
        
        for channel in self.cfg.SENSORS:
            sample={}
            sd_record = self.get('sample_data', sample_record['data'][channel])
            sample['channel']=channel
            sample['token']=sd_record['token']
            sample['filename']=sd_record['filename']
            if 'CAM' in channel:
                frame['camera'].append(sample)
            elif 'RADAR' in channel:
                frame['radar'].append(sample)
            elif 'LIDAR' in channel:
                frame['lidar'].append(sample)
            else:
                raise Exception('Channel not recognized.')
        ## if 'one_cam' option is chosen, create as many frames as cameras
        if self.cfg.SAMPLE_MODE == "one_cam":
            for cam in frame['camera']:
                temp_frame = copy.deepcopy(frame)
                temp_frame['camera']=[cam]
                temp_frame['id'] = self.frame_id
                frames.append(temp_frame)
                self.frame_id += 1
        ## if 'all_cam' option is chosen, create one frame with all camera images
        elif self.cfg.SAMPLE_MODE == "all_cam":
            frame['id'] = self.frame_id
            frames.append(frame)
            self.frame_id += 1

        return frames
    ##--------------------------------------------------------------------------
    def _get_sensor_data(self, idx):
        """
        Returns a frame with sensor data in vehicle or global coordinates
        
        :param idx (int): id of the dataset's split to retrieve
        :return sensor_data (dict): all sensor data for that frame
        """
        frame = copy.deepcopy(self.db['frames'][idx])      
        sample_record = self.get('sample', frame['sample_token'])
        
        ## Get camera data if requested
        for i, cam in enumerate(frame['camera']):
            image, cs_record, pose_record, filename = self._get_cam_data(cam['token'])
            frame['camera'][i]['image'] = image
            frame['camera'][i]['cs_record'] = cs_record
            frame['camera'][i]['pose_record'] = pose_record
            frame['camera'][i]['filename'] = filename
            frame['camera'][i]['img_id'] = self.img_id
            self.img_id += 1

        ## Get LIDAR data if requested
        for i, lidar in enumerate(frame['lidar']):
            lidar_pc, lidar_cs, pose_record = self._get_pointsensor_data('lidar',
                                                            sample_record,
                                                            lidar['token'],
                                                            self.cfg.LIDAR_SWEEPS)
            frame['lidar'][i]['pointcloud'] = lidar_pc
            frame['lidar'][i]['cs_record'] = lidar_cs
            frame['lidar'][i]['pose_record'] = pose_record

        ## Get Radar data if requested
        if len(frame['radar']) > 0:
            all_radar_pcs = RadarPointCloud(np.zeros((18, 0)))
            for i, radar in enumerate(frame['radar']):
                radar_pc, _ , pose_record = self._get_pointsensor_data('radar',
                                                            sample_record, 
                                                            radar['token'], 
                                                            self.cfg.RADAR_SWEEPS)
                all_radar_pcs.points = np.hstack((all_radar_pcs.points, radar_pc.points))
            ## TODO: since different Radar point clouds are merged, 
            ## pose_record for the last Radar is saved as the pose_record.
            frame['radar'] = [{'pointcloud':all_radar_pcs, 
                            'pose_record': pose_record}]
        
        ## Get annotations using the LIDAR token
        lidar_token = sample_record['data']['LIDAR_TOP']
        frame['anns'] = self._get_annotations(lidar_token)
        
        return frame
    ##--------------------------------------------------------------------------
    def _get_cam_data(self, cam_token):
        """
        Get camera sample data

        :param cam_token (str): sample data token for this camera
        :return image, intrinsics, pose_record, path:
        """
        filename = self.get_sample_data_path(cam_token)
        cam_record = self.get('sample_data', cam_token)
        cs_record = self.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        pose_record = self.get('ego_pose', cam_record['ego_pose_token'])
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                image_str = f.read()
        else:
            raise Exception('Camera image not found at {}'.format(filename))
        image = np.array(Image.open(io.BytesIO(image_str)))
        return image, cs_record, pose_record, filename
    ##--------------------------------------------------------------------------
    def _get_pointsensor_data(self, sensor_type, sample_record, sensor_token, nsweeps=1):
        """
        Returns the LIDAR pointcloud for this frame in vehicle/global coordniates
        
        :param sensor_type (str): 'radar' or 'lidar'
        :param sample_record: sample record dictionary from nuscenes
        :param sensor_token: sensor data token
        :param nsweeps: number of previous sweeps to include
        :return pc, cs_record, pose_record: Point cloud and other sensor parameters
        """
        sd_record = self.get('sample_data', sensor_token)
        pose_record = self.get('ego_pose', sd_record['ego_pose_token'])
        cs_record = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        
        ## Read data from file (in sensor's coordinates)
        if sensor_type == 'lidar':
            pc, _ = LidarPointCloud.from_file_multisweep(self,
                                                        sample_record, 
                                                        sd_record['channel'], 
                                                        sd_record['channel'], 
                                                        nsweeps=nsweeps,
                                                        min_distance=self.cfg.PC_MIN_DIST)
        elif sensor_type == 'radar':
            pc, _ = RadarPointCloud.from_file_multisweep(self,
                                                        sample_record, 
                                                        sd_record['channel'], 
                                                        sd_record['channel'], 
                                                        nsweeps=nsweeps,
                                                        min_distance=self.cfg.PC_MIN_DIST)
        else:
            raise Exception('Sensor type not valid.')
        
        ## Take point clouds from sensor to vehicle coordinates
        pc = nsutils.sensor_to_vehicle(pc, cs_record)       
        if self.cfg.COORDINATES == 'global':
            ## Take point clouds from vehicle to global coordinates
            pc = nsutils.vehicle_to_global(pc, pose_record)
        return pc, cs_record, pose_record
    ##--------------------------------------------------------------------------
    def _get_annotations(self, sample_data_token):
        """
        Gets the annotations for this sample in the vehicle coordinates
        
        :param sample_data_token: Unique sample_data identifier
        :return box_list (Box): list of Nuscenes Boxes
        """
        if 'test' in self.cfg.SPLIT:
            return []
    
        box_list = []
        sd_record = self.get('sample_data', sample_data_token)
        pose_record = self.get('ego_pose', sd_record['ego_pose_token'])
        
        ## Get boxes from nuscenes (boxes are in global coordinates)
        orig_box_list = self.get_boxes(sample_data_token)
        for box in orig_box_list:
            if box.name not in self.cfg.CLASSES:
                continue
            if self.cfg.MAX_BOX_DIST:
                box_dist = nsutils.get_box_dist(box, pose_record)
                if box_dist > abs(self.cfg.MAX_BOX_DIST):
                    continue
            
            if self.cfg.BOX_VELOCITY:
                box.velocity = self.box_velocity(box.token)
            ## Global to Vehicle
            if self.cfg.COORDINATES == 'vehicle':
                box = nsutils.global_to_vehicle(box, pose_record)
            box_list.append(box)
        return box_list
    ##--------------------------------------------------------------------------
    def _get_sweep_tokens(self, sd_record):
        """
        Get previous sensor sweeps for the given sample record token
        
        :param sd_record (dict): sensor record 
        :return sweeps (list): list of sweeps for the sensor sample
        """
        sweeps = []
        ind = 0
        sensor_name = sd_record['channel']
        if 'CAM' in sensor_name:
            n_sweeps = self.cfg.CAMERA_SWEEPS
        elif 'RADAR' in sd_record['channel']:
            n_sweeps = self.cfg.RADAR_SWEEPS
        elif 'LIDAR' in sd_record['channel']:
            n_sweeps = self.cfg.LIDAR_SWEEPS
        else:
            raise Exception('This should not happen!')
            
        while ind < n_sweeps:
            if not sd_record['prev'] == "":
                sweeps.append(sd_record['prev'])
                sd_record = self.get('sample_data', sd_record['prev'])
            else:
                break
        return sweeps
##------------------------------------------------------------------------------
if __name__ == "__main__":
    nusc_ds = NuscenesDataset(cfg='config/cfg.yml')
    print(nusc_ds[0])