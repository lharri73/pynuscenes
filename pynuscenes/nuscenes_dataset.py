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

        ## Sanity checks
        assert self.cfg.COORDINATES in ['vehicle', 'global'], \
            'COORDINATES system not valid.'
        assert self.cfg.SPLIT in _C.NUSCENES_SPLITS[self.cfg.VERSION], \
            'SPLIT not valid.'
        assert self.cfg.SAMPLE_MODE in ["all_cam", "one_cam"], \
            'SAMPLE_MODE not valid.'

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
        frame = copy.deepcopy(self.db['frames'][idx])
        data = self.dataset_mapper(frame)
        return data
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
        ## TODO: load a pickled db if it exists
        startTime = time.time()
        scenes_list = nsutils.split_scenes(self.scene, self.cfg.SPLIT)
        self.logger.info('Scenes in {} split: {}'.format(self.cfg.SPLIT, 
                                                        str(len(scenes_list))))

        ## Loop over all the scenes
        self.logger.info('Creating database')
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
        Create frames from a single sample from the nuscenes dataset
        
        :param sample_record (dict): sample record dictionary from nuscenes
        :return frames (list): list of frame dictionaries
        """
        frame = {'coordinates': self.cfg.COORDINATES,
                 'sample_token': sample_record['token'],
                 'anns':[]}
        all_frames = []
        camera = []
        radar = []
        
        ## Get sensor info
        for channel in self.cfg.SENSORS:
            sd_record = self.get('sample_data', sample_record['data'][channel])
            sample={'channel': channel,
                    'token': sd_record['token'],
                    'filename': sd_record['filename']}
            if 'CAM' in channel:
                camera.append(sample)
            elif 'RADAR' in channel:
                radar.append(sample)
            elif 'LIDAR' in channel:
                lidar = sample
            else:
                raise Exception('Channel not recognized.')
        
        ## Add sensors to the frame dictionary
        if len(camera): frame['camera']=camera
        if len(radar): frame['radar']=radar
        if len(lidar): frame['lidar']=lidar

        ## Add annotation tokens for select categories
        ## TODO: add filtering based on num_radar/num_lidar points here
        ann_tokens = sample_record['anns']
        for token in ann_tokens:
            ann = self.get('sample_annotation', token)
            if ann['category_name'] in self.cfg.CATEGORIES:
                frame['anns'].append(ann)

        ## if 'one_cam' option is chosen, create as many frames as there are cameras
        if 'camera' in frame and self.cfg.SAMPLE_MODE == "one_cam":
            for cam in frame['camera']:
                temp_frame = copy.deepcopy(frame)
                temp_frame['camera']=[cam]
                temp_frame['id'] = self.frame_id
                all_frames.append(temp_frame)
                self.frame_id += 1
        else:
            ## create one frame with all camera images
            frame['id'] = self.frame_id
            all_frames.append(frame)
            self.frame_id += 1

        return all_frames
    ##--------------------------------------------------------------------------
    def dataset_mapper(self, frame):
        """
        Returns a frame with sensor data in vehicle or global coordinates
        
        :param frame (dict): A frame dictionary from the db (no sensor data)
        :return frame (dict): frame with all sensor data
        """
        sample_record = self.get('sample', frame['sample_token'])

        ## Get camera data
        if 'camera' in frame:
            for i, cam in enumerate(frame['camera']):
                image, cs_record, pose_record, filename = self._get_cam_data(cam['token'])
                frame['camera'][i]['image'] = image
                frame['camera'][i]['cs_record'] = cs_record
                frame['camera'][i]['pose_record'] = pose_record
                frame['camera'][i]['filename'] = filename
                frame['camera'][i]['img_id'] = self.img_id
                self.img_id += 1

        ## Get LIDAR data
        if 'lidar' in frame:
            lidar_pc, lidar_cs, lidar_pose_record = self._get_pointsensor_data('lidar',
                                                            sample_record,
                                                            frame['lidar']['token'],
                                                            self.cfg.LIDAR_SWEEPS)
            
            ## Filter points outside the image
            if self.cfg.FILTER_PC and self.cfg.SAMPLE_MODE == "one_cam":
                cam = frame['camera'][0]
                _, _, mask = nsutils.map_pointcloud_to_image(lidar_pc,
                                        cam_cs_record=cam['cs_record'],
                                        cam_pose_record=cam['pose_record'],
                                        img_shape=(1600,900),
                                        pointsensor_pose_record=lidar_pose_record,
                                        coordinates=frame['coordinates'])
                lidar_pc.points = lidar_pc.points[:,mask]
            
            frame['lidar']['pointcloud'] = lidar_pc
            frame['lidar']['cs_record'] = lidar_cs
            frame['lidar']['pose_record'] = lidar_pose_record

        ## Get Radar data
        if 'radar' in frame:
            all_radar_pcs = RadarPointCloud(np.zeros((18, 0)))
            for i, radar in enumerate(frame['radar']):
                radar_pc, _ , radar_pose_record = self._get_pointsensor_data('radar',
                                                        sample_record, 
                                                        radar['token'], 
                                                        self.cfg.RADAR_SWEEPS)
            
                ## Filter points outside the image
                if self.cfg.FILTER_PC and self.cfg.SAMPLE_MODE == "one_cam":
                    cam = frame['camera'][0]
                    _, _, mask = nsutils.map_pointcloud_to_image(radar_pc,
                                            cam_cs_record=cam['cs_record'],
                                            cam_pose_record=cam['pose_record'],
                                            pointsensor_pose_record=radar_pose_record,
                                            coordinates=frame['coordinates'])
                    radar_pc.points = radar_pc.points[:,mask]
                
                all_radar_pcs.points = np.hstack((all_radar_pcs.points, radar_pc.points))

            ## TODO: pose_record for the last Radar is saved as the pose_record.
            frame['radar'] = {}
            frame['radar']['pointcloud'] = all_radar_pcs
            frame['radar']['pose_record'] = radar_pose_record
        
        ## Get all annotations
        ann_records = frame['anns']
        ref_token = sample_record['data'][self.cfg.ANN_REF_FRAME]
        ref_sd_record = self.get('sample_data', ref_token)
        ref_pose_record = self.get('ego_pose', ref_sd_record['ego_pose_token'])
        frame['ref_pose_record'] = ref_pose_record
        
        all_anns = self._get_anns(ann_records, ref_pose_record)
        frame['anns'] = all_anns

        ## Filter annotations
        if self.cfg.FILTER_ANNS and self.cfg.SAMPLE_MODE == "one_cam":
            assert len(frame['camera']) == 1, \
                'More than one camera available for filtering.'
            frame['anns'] = []
        
            cam_cs_record = frame['camera'][0]['cs_record']
            camera_intrinsic = np.array(cam_cs_record['camera_intrinsic'])
            lid_sd_record = self.get('sample_data', sample_record['data']['LIDAR_TOP'])
            ref_pose_record = self.get('ego_pose', lid_sd_record['ego_pose_token'])
            
            for i, box in enumerate(all_anns):
                box_cam = copy.deepcopy(box)
                ## Go to camera coordinates
                if frame['coordinates'] == 'global':
                    box_cam = nsutils.global_to_vehicle(box_cam, ref_pose_record)
                box_cam = nsutils.vehicle_to_sensor(box_cam, cam_cs_record)
                if box_in_image(box_cam, camera_intrinsic, (1600,900)):
                    frame['anns'].append(box)

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
    def _get_anns(self, ann_records, ref_pose_record):
        """
        Get the annotations in Box format for this sample
        
        :param ann_records: annotation records
        :return box_list (Box): list of Nuscenes Boxes
        """

        boxes = []
        sample_token = ann_records[0]['sample_token']
        sample_record = self.get('sample', sample_token)
        
        ## Get boxes (boxes are in global coordinates)
        for ann in ann_records:
            box = self.get_box(ann['token'])
            box.name = self.cfg.CATEGORIES[box.name]

            ## Filter based on distance to vehicle
            if self.cfg.MAX_BOX_DIST:
                box_dist = nsutils.get_box_dist(box, ref_pose_record)
                if box_dist > abs(self.cfg.MAX_BOX_DIST):
                    continue
            
            ## Calculate box velocity
            if self.cfg.BOX_VELOCITY:
                box.velocity = self.box_velocity(box.token)

            ## Global to Vehicle
            if self.cfg.COORDINATES == 'vehicle':
                box = nsutils.global_to_vehicle(box, ref_pose_record)

            boxes.append(box)     

        return boxes
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