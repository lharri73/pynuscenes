import io
import os
import cv2
import time
import copy
import pickle
import numpy as np
from PIL import Image
import os.path as osp
from pprint import pprint
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
        self.image_id = 0

        ## Sanity checks
        assert self.cfg.COORDINATES in ['vehicle', 'global'], \
            'COORDINATES system not valid.'
        assert self.cfg.SPLIT in _C.NUSCENES_SPLITS[self.cfg.VERSION], \
            'SPLIT not valid.'
        assert self.cfg.SAMPLE_MODE in ["camera", "scene"], \
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
    def get_categories(self):
        """
        Return dataset categories and their IDs
        """
        categories = {}
        for cat in self.cfg.CATEGORIES.values():
            categories[cat] = self.cfg.CAT_ID[cat]
        return categories
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
        Create frames from a single sample of the nuscenes dataset
        
        :param sample_record (dict): sample record dictionary from nuscenes API
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
            # cs_record = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            # pose_record = self.get('ego_pose', sd_record['ego_pose_token'])
            sample={'channel': channel,
                    # 'cs_record': cs_record,
                    # 'pose_record': pose_record,
                    'sd_record': sd_record,
                    # 'token': sd_record['token'],
                    'filename': osp.join(self.dataroot, sd_record['filename']),
            }
            if 'CAM' in channel:
                sample['width'] = sd_record['width']
                sample['height'] = sd_record['height']
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

        ## if 'one_cam' option is chosen, create as many frames as there are cameras
        if 'camera' in frame and self.cfg.SAMPLE_MODE == "camera":
            for cam in frame['camera']:
                temp_frame = copy.deepcopy(frame)
                temp_frame['camera']=[cam]
                temp_frame['id'] = self.frame_id
                all_frames.append(temp_frame)
                self.frame_id += 1
        else:
            ## create one frame with all data
            frame['id'] = self.frame_id
            all_frames.append(frame)
            self.frame_id += 1

        return all_frames
    ##--------------------------------------------------------------------------
    def _get_anns(self, ann_tokens, cam=None):
        """
        Get all annotations for a given sample
        :param sample_record (dict):
        :param ref_channel (str):
        """        
        ## TODO: add filtering based on num_radar/num_lidar points here
        ref_pose_record = None
        if cam is not None:
            ref_cs_record = cam['cs_record']
            ref_pose_record = cam['pose_record']
            cam_intrinsic = np.array(ref_cs_record['camera_intrinsic'])

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

            ## Create Box object (boxes are in global coordinates)
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']),
                    name=this_ann['category'], token=ann['token'])

            ## Filter for camera visibility
            if cam is not None:
                box_veh = nsutils.global_to_vehicle(box, ref_pose_record)
                box_cam = nsutils.vehicle_to_sensor(box_veh, ref_cs_record)
                if not box_in_image(box_cam, cam_intrinsic, (cam['width'], cam['height'])):
                    continue
                
                ## Get distance to camera
                box_dist = nsutils.get_box_dist(box, ref_pose_record)
                this_ann['distance'] = box_dist
                # box = box_cam
            
                if self.cfg.MAX_BOX_DIST:
                    if box_dist > abs(self.cfg.MAX_BOX_DIST):
                        continue
            
            ## Calculate box velocity
            if self.cfg.BOX_VELOCITY:
                box.velocity = self.box_velocity(box.token)

            ## Take to the right coordinate system
            if self.cfg.COORDINATES == 'vehicle':
                box = nsutils.global_to_vehicle(box, ref_pose_record)

            this_ann['box_3d'] = box
            annotations.append(this_ann)
        return annotations, ref_pose_record
    ##--------------------------------------------------------------------------
    def dataset_mapper(self, frame):
        """
        Add sensor data in vehicle or global coordinates to the frame
        
        :param frame (dict): A frame dictionary from the db (no sensor data)
        :return frame (dict): frame with all sensor data
        """
        sample_record = self.get('sample', frame['sample_token'])

        ## Load camera data
        if 'camera' in frame:
            for i, cam in enumerate(frame['camera']):
                image = self._get_camera_data(cam['filename'])
                sd_record = cam.pop('sd_record')
                cam['image'] = image
                cam['cs_record'] = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                cam['pose_record'] = self.get('ego_pose', sd_record['ego_pose_token'])
        
        ## Load annotations
        if self.cfg.SAMPLE_MODE == 'camera':
            anns, ref_pose_rec = self._get_anns(frame['anns'], frame['camera'][0])
        else:
            anns, ref_pose_rec = self._get_anns(frame['anns'])
        frame['anns'] = anns
        frame['ref_pose_record'] = ref_pose_rec

        ## Load LIDAR data
        if 'lidar' in frame:
            lidar = frame['lidar']
            sd_record = lidar.pop('sd_record')
            lidar['cs_record'] = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            lidar['pose_record'] = self.get('ego_pose', sd_record['ego_pose_token'])
            lidar_pc = self._get_pointsensor_data('lidar',
                                                sample_record,
                                                lidar['channel'],
                                                lidar['cs_record'],
                                                lidar['pose_record'],
                                                nsweeps=self.cfg.LIDAR_SWEEPS)
            
            ## Filter points outside the image
            if self.cfg.FILTER_PC and self.cfg.SAMPLE_MODE == "camera":
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

        ## Load Radar data
        if 'radar' in frame:
            all_radar_pcs = RadarPointCloud(np.zeros((18, 0)))
            for i, radar in enumerate(frame['radar']):
                sd_record = radar.pop('sd_record')
                radar['cs_record'] = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                radar['pose_record'] = self.get('ego_pose', sd_record['ego_pose_token'])
                radar_pc = self._get_pointsensor_data('radar',
                                                sample_record, 
                                                radar['channel'], 
                                                radar['cs_record'],
                                                radar['pose_record'],
                                                nsweeps=self.cfg.RADAR_SWEEPS)
            
                ## Filter points outside the image
                if self.cfg.FILTER_PC and self.cfg.SAMPLE_MODE == "camera":
                    cam = frame['camera'][0]
                    cam_intrinsic = np.array(cam['cs_record']['camera_intrinsic'])
                    radar_pc_cam, _ = nsutils.map_pointcloud_to_camera(radar_pc, 
                                                cam_cs_record=cam['cs_record'],
                                                cam_pose_record=cam['pose_record'],
                                                pointsensor_pose_record=radar['pose_record'],
                                                coordinates=frame['coordinates'])
                    _, _, mask = nsutils.map_pointcloud_to_image(radar_pc_cam, 
                                                                cam_intrinsic,
                                                                img_shape=(1600,900))               
                    radar_pc.points = radar_pc.points[:,mask]
                
                all_radar_pcs.points = np.hstack((all_radar_pcs.points, radar_pc.points))

            ## TODO: pose_record for the last Radar is saved as the pose_record.
            frame['radar'] = {}
            frame['radar']['pointcloud'] = all_radar_pcs
            frame['radar']['pose_record'] = radar['pose_record']

        return frame
    ##--------------------------------------------------------------------------
    def _get_camera_data(self, filename):
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
    ##--------------------------------------------------------------------------
    def _get_pointsensor_data(self, sensor_type, sample_record, channel, 
                              cs_record, pose_record, nsweeps=1):
        """
        Returns the LIDAR pointcloud for this frame in vehicle/global coordniates
        
        :param sensor_type (str): 'radar' or 'lidar'
        :param sample_record: sample record dictionary from nuscenes
        :param channel: sensor channel
        :param nsweeps: number of previous sweeps to include
        :return pc, cs_record, pose_record: Point cloud and other sensor parameters
        """
        
        ## Read data from file (in sensor's coordinates)
        if sensor_type == 'lidar':
            pc, _ = LidarPointCloud.from_file_multisweep(self,
                                                        sample_record, 
                                                        channel, 
                                                        channel, 
                                                        nsweeps=nsweeps,
                                                        min_distance=self.cfg.PC_MIN_DIST)
        elif sensor_type == 'radar':
            pc, _ = RadarPointCloud.from_file_multisweep(self,
                                                        sample_record, 
                                                        channel, 
                                                        channel, 
                                                        nsweeps=nsweeps,
                                                        min_distance=self.cfg.PC_MIN_DIST)
        else:
            raise Exception('Sensor type not valid.')
        
        ## Take point clouds from sensor to vehicle coordinates
        pc = nsutils.sensor_to_vehicle(pc, cs_record)       
        if self.cfg.COORDINATES == 'global':
            ## Take point clouds from vehicle to global coordinates
            pc = nsutils.vehicle_to_global(pc, pose_record)
        
        return pc
##------------------------------------------------------------------------------
if __name__ == "__main__":
    nusc_ds = NuscenesDataset(cfg='config/cfg.yml')
    print(nusc_ds[0])