import io
import os
import time
import copy
import pickle
from easydict import EasyDict
import numpy as np
from PIL import Image
import os.path as osp
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import box_in_image
from nuscenes.utils.data_classes import Box, LidarPointCloud, RadarPointCloud

import pynuscenes.utils.nuscenes_utils as nsutils
from pyquaternion import Quaternion
from pynuscenes.utils import constants as _C
from pynuscenes.utils.io_utils import yaml_load
from pynuscenes.utils import log
from pprint import pprint


class NuscenesDataset(NuScenes):
    """
    An improved database and dataloader class for nuScenes.
    """
    def __init__(self, dataroot, version, split, cfg, generate_db=True):
        """
        :param cfg (str): path to the config file
        """
        self.cfg = cfg.NUSCENES if isinstance(cfg, EasyDict) else yaml_load(cfg, safe_load=True).NUSCENES
        ## Sanity checks
        assert split in _C.NUSCENES_SPLITS[version], \
            'SPLIT not valid.'
        assert self.cfg.SAMPLE_MODE in ["camera", "scene"], \
            'SAMPLE_MODE not valid.'
        if self.cfg.SAMPLE_MODE == "camera":
            assert sum([1 for key in self.cfg.SENSORS if 'CAM' in key]), \
                'At least one camera should be in SENSORS when "camera" sample mode is selected.'
        
        self.logger = log.getLogger(__name__, console_level=self.cfg.CONSOLE_LOG_LEVEL)
        self.dataroot = dataroot
        self.frame_id = 0
        self.image_id = 0
        self.cfg.COORDINATES = 'vehicle'
        self.logger.info('Loading NuScenes')
        self.cfg.VERSION = version
        self.cfg.SPLIT = split

        self.categories = {}

        ## Create dataset categories
        if "CAT_ID" in self.cfg:
            self.categories = {k:v for k,v in self.cfg.CAT_ID.items() if k in self.cfg.CATEGORIES.values()}
        else:
            self.categories = {k:v for k,v in _C.DETECTION_ID.items() if k in self.cfg.CATEGORIES.values()}
        self.categories = {k:v for k,v in sorted(self.categories.items(), key=lambda x: x[1])}

        super().__init__(version = self.cfg.VERSION,
                         dataroot = self.dataroot,
                         verbose = self.cfg.VERBOSE)
        self.db = {}
        if generate_db:
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
    def get_categories(self):
        """
        Return dataset categories and their IDs
        """
        return self.categories
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
        for scene in scenes_list:
            scene_frames = []
            scene_record= self.get('scene', scene)
            sample_record= self.get('sample', scene_record['first_sample_token'])
            
            ## Loop over all samples in this scene
            while True:
                scene_frames += self._get_sample_frames(sample_record)
                if sample_record['next'] == "":
                    break
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
        Create frames from a single sample of the nuscenes dataset
        
        :param sample_record (dict): sample record dictionary from nuscenes API
        :return frames (list): list of frame dictionaries
        """
        frame = {'coordinates': self.cfg.COORDINATES,
                 'sample_token': sample_record['token']}
        all_frames = []
        camera = []
        radar = []
        
        ## Get sensor info
        for channel in self.cfg.SENSORS:
            # sd_record = self.get('sample_data', sample_record['data'][channel])
            sample={'channel': channel,
                    'token': sample_record['data'][channel],
            }
            if 'CAM' in channel:
                sample['image_id'] = self.image_id
                self.image_id += 1
                camera.append(sample)
            
            elif 'RADAR' in channel:
                radar.append(sample)
            
            elif 'LIDAR' in channel:
                frame['lidar']=sample
            
            else:
                raise Exception('Channel not recognized.')
        
        ## Add sensors to the frame dictionary (LIDAR is already added)
        if len(camera): frame['camera']=camera
        if len(radar): frame['radar']=radar

        ## Get annotations
        frame['anns'] = sample_record['anns']
        
        ## for 'camera' sample option, create one frame for each camera
        if 'camera' in frame and self.cfg.SAMPLE_MODE == "camera":
            for cam in frame['camera']:
                temp_frame = copy.deepcopy(frame)
                temp_frame['camera']=[cam]
                temp_frame['id'] = self.frame_id

                if self.cfg.FILTER_RADARS:
                    ## Filter Radars based on camera view
                    temp_frame['radar'] = [x for x in temp_frame['radar'] if 
                        x['channel'] in _C.RADAR_FOR_CAMERA[cam['channel']]]
                
                all_frames.append(temp_frame)
                self.frame_id += 1
            
        else:
            ## for 'scene' sample option, create one frame for all sensors
            frame['id'] = self.frame_id
            all_frames.append(frame)
            self.frame_id += 1

        return all_frames
    ##--------------------------------------------------------------------------
    def _get_anns(self, ann_tokens, ref_pose_record):
        """
        Get all annotations for a given sample
        :param ann_tokens (list): list of annotation tokens
        :param cam (dict): ca
        """        
        ## TODO: add filtering based on num_radar/num_lidar points here
        ## TODO: for 'scene' samples, ref_pose_record stuff needs attention here

        annotations = []
        for i, token in enumerate(ann_tokens):
            ann = self.get('sample_annotation', token)
            this_ann = {}
            try:
                this_ann['category'] = self.cfg.CATEGORIES[ann['category_name']]
            except:
                continue
            this_ann['category_id'] = self.cfg.CAT_ID[this_ann['category']]
            this_ann['num_lidar_pts'] = ann['num_lidar_pts']
            this_ann['num_radar_pts'] = ann['num_radar_pts']
            this_ann['instance_token'] = ann['instance_token']

            ## Create Box object (boxes are in global coordinates)
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']),
                    name=this_ann['category'], token=ann['token'])
            
            ## Get distance to vehicle
            box_dist = nsutils.get_box_dist(box, ref_pose_record)
            this_ann['distance'] = box_dist
            if self.cfg.MAX_BOX_DIST:
                if box_dist > abs(self.cfg.MAX_BOX_DIST):
                    continue

            ## Take to the vehicle coordinate system
            box = nsutils.global_to_vehicle(box, ref_pose_record)
            
            ## Calculate box velocity
            if self.cfg.BOX_VELOCITY:
                box.velocity = self.box_velocity(box.token)

            this_ann['box_3d'] = box
            annotations.append(this_ann)
        return annotations
    ##--------------------------------------------------------------------------
    def dataset_mapper(self, frame):
        """
        Add sensor data in vehicle or global coordinates to the frame
        
        :param frame (dict): A frame dictionary from the db (no sensor data)
        :return frame (dict): frame with all sensor data
        """
        sample_record = self.get('sample', frame['sample_token'])
        ref_sd_record = self.get('sample_data', sample_record['data'][self.cfg.REF_POSE_CHANNEL])
        ref_pose_record = self.get('ego_pose', ref_sd_record['ego_pose_token'])
        ref_cs_record = self.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
        frame['ref_pose_record'] = ref_pose_record

        ## Load camera data
        cams = frame.get('camera',[])
        for cam in cams:
            sd_record = self.get('sample_data', cam['token'])
            filename = osp.join(self.dataroot, sd_record['filename'])
            image = self.get_camera_data(filename)
            cam['image'] = image
            cam['height'] = sd_record['height']
            cam['width'] = sd_record['width']
            cam['filename'] = filename
            cam['cs_record'] = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            cam['pose_record'] = self.get('ego_pose', sd_record['ego_pose_token'])
        
        ## Load annotations in vehicle coordinates
        frame['anns'] = self._get_anns(frame['anns'],  
                                       ref_pose_record)
        
        ## Filter anns outside image if in 'camera' mode:
        if self.cfg.SAMPLE_MODE == 'camera':
            filtered_anns = []
            for ann in frame['anns']:
                box_cam = nsutils.map_annotation_to_camera(ann['box_3d'], 
                                                           cam['cs_record'],
                                                           cam['pose_record'],
                                                           ref_pose_record,
                                                           self.cfg.COORDINATES)
                cam_intrinsic = np.array(cam['cs_record']['camera_intrinsic'])
                if not box_in_image(box_cam, cam_intrinsic, (1600, 900)):
                    continue
                if self.cfg.GEN_2D_BBOX:
                    ann['box_2d'] = nsutils.box_3d_to_2d_simple(box_cam, 
                                                                cam_intrinsic, 
                                                                (1600, 900))
                filtered_anns.append(ann)
            frame['anns'] = filtered_anns

        ## Load LIDAR data in vehicle coordinates
        if 'lidar' in frame:
            lidar = frame['lidar']
            sd_record = self.get('sample_data', lidar['token'])
            lidar['cs_record'] = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            lidar['pose_record'] = self.get('ego_pose', sd_record['ego_pose_token'])
            lidar_pc, _ = LidarPointCloud.from_file_multisweep(self,
                                                        sample_record, 
                                                        lidar['channel'], 
                                                        lidar['channel'], 
                                                        nsweeps=self.cfg.LIDAR_SWEEPS,
                                                        min_distance=self.cfg.PC_MIN_DIST)
            ## Take from sensor to vehicle coordinates
            lidar_pc = nsutils.sensor_to_vehicle(lidar_pc, lidar['cs_record'])   
            
            ## filter points outside the image if in 'camera' mode
            if self.cfg.SAMPLE_MODE == "camera":
                cam = frame['camera'][0]
                cam_intrinsic = np.array(cam['cs_record']['camera_intrinsic'])
                lidar_pc_cam, _ = nsutils.map_pointcloud_to_camera(lidar_pc, 
                                            cam_cs_record=cam['cs_record'],
                                            cam_pose_record=cam['pose_record'],
                                            pointsensor_pose_record=frame['lidar']['pose_record'],
                                            coordinates=frame['coordinates'])
                _, _, mask = nsutils.map_pointcloud_to_image(lidar_pc_cam, 
                                                             cam_intrinsic,
                                                             img_shape=(1600,900))               
                lidar_pc.points = lidar_pc.points[:,mask]
            frame['lidar']['pointcloud'] = lidar_pc

        ## Load Radar data in vehicle coordinates
        if 'radar' in frame:
            all_radar_pcs = RadarPointCloud(np.zeros((18, 0)))
            for radar in frame['radar']:
                radar_pc, _ = RadarPointCloud.from_file_multisweep(self,
                                                        sample_record, 
                                                        radar['channel'], 
                                                        self.cfg.REF_POSE_CHANNEL, 
                                                        nsweeps=self.cfg.RADAR_SWEEPS,
                                                        min_distance=self.cfg.PC_MIN_DIST)
                radar_pc = nsutils.sensor_to_vehicle(radar_pc, ref_cs_record) 

                ## filter points outside the image if in 'camera' mode
                if self.cfg.SAMPLE_MODE == "camera":
                    cam = frame['camera'][0]
                    cam_intrinsic = np.array(cam['cs_record']['camera_intrinsic'])
                    radar_pc_cam, _ = nsutils.map_pointcloud_to_camera(radar_pc, 
                                                cam_cs_record=cam['cs_record'],
                                                cam_pose_record=cam['pose_record'],
                                                pointsensor_pose_record=ref_pose_record,
                                                coordinates=frame['coordinates'])
                    _, _, mask = nsutils.map_pointcloud_to_image(radar_pc_cam, 
                                                                cam_intrinsic,
                                                                img_shape=(1600,900))               
                    radar_pc.points = radar_pc.points[:,mask]
                
                all_radar_pcs.points = np.hstack((all_radar_pcs.points, radar_pc.points))

            frame['radar'] = {}
            frame['radar']['pointcloud'] = all_radar_pcs
            frame['radar']['pose_record'] = ref_pose_record
            frame['radar']['cs_record'] = ref_cs_record

        return frame
    ##--------------------------------------------------------------------------
    @staticmethod
    def get_camera_data(filename):
        """
        Get camera image

        :param filename (str): image file path
        :return image (Image)
        """
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                image_str = f.read()
        else:
            raise Exception('Camera image not found at {}'.format(filename))
        image = np.array(Image.open(io.BytesIO(image_str)))
        return image
##------------------------------------------------------------------------------
if __name__ == "__main__":
    nusc_ds = NuscenesDataset(cfg='config/cfg.yml')
    print(nusc_ds[0])
