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
    def __init__(self, cfg) -> None:
        """
        :param cfg (str): path to the config file
        """
        self.cfg = io_utils.yaml_load(cfg, safe_load=True)
        self.logger = log.getLogger(__name__)
        self.logger.info('Loading NuScenes')
        self.frame_id = 0

        assert self.cfg.COORDINATES in ['vehicle', 'global'], \
            'Coordinate system not valid.'
        assert self.cfg.SPLIT in _C.NUSCENES_SPLITS[self.cfg.VERSION], \
            'NuScenes split not valid.'

        super().__init__(version = self.cfg.VERSION,
                         dataroot = self.cfg.DATA_ROOT,
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
    def generate_db(self, out_dir=None) -> dict:
        """
        Read and preprocess the dataset samples and annotations, representing 
        the dataset items in a lightweight, canonical format. This function does 
        not read the sensor data files (e.g., images are not loaded into memory).

        :param out_dir (str): Directory to save the database pickle file
        :returns dataset_dicts (dict): a dictionary containing dataset meta-data 
            and a list of dicts, one for each sample in the dataset
        """
        startTime = time.time()
        scenes_list = nsutils.split_scenes(self.scene, self.cfg.SPLIT)
        self.logger.info('Scenes in {} split: {}'.format(self.cfg.SPLIT, 
                                                        str(len(scenes_list))))
        self.logger.info('Creating database')
        # frames = self._get_frames(scenes_list)
        all_frames = []
        for scene in tqdm(scenes_list):
            scene_frames = []
            scene_rec = self.get('scene', scene)
            sample_rec = self.get('sample', scene_rec['first_sample_token'])
            ## Loop over all samples in the scene
            has_more_samples = True
            while has_more_samples:
                scene_frames += self._get_sample_frames(sample_rec)
                if sample_rec['next'] == "":
                    has_more_samples = False
                else:
                    sample_rec = self.get('sample', sample_rec['next'])
            all_frames += scene_frames

        metadata = {"version": self.cfg.VERSION}
        db = {"frames": all_frames,
              "metadata": metadata}
        
        self.logger.info('Created database in %.1f seconds' % (time.time()-startTime))
        self.logger.info('Samples in {} split: {}'.format(self.cfg.SPLIT,
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
    def _get_sample_frames(self, sample_rec):
        """
        Get all the frames from a single sample from the nuscenes dataset
        
        :param sample_rec (dict): sample record dictionary from nuscenes
        :return frames (list): list of frame dictionaries
        """
        frames = []
        frame = {
            'camera':[],
            'lidar': [],
            'radar':[],
        }
        ## Generate a frame containing all sensors
        frame['anns'] = sample_rec['anns']
        for channel in self.cfg.SENSORS:
            sample={}
            sensor_rec = self.get('sample_data', sample_rec['data'][channel])
            sample['channel']=channel
            sample['token']=sensor_rec['token']
            sample['filename']=sensor_rec['filename']
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
    def _get_sweep_tokens(self, sensor_record) -> dict:
        """
        Get previous sensor sweeps for the given sample record token
        
        :param sensor_record (dict): sensor record 
        :return sweeps (list): list of sweeps for the sensor sample
        """
        sweeps = []
        ind = 0
        sensor_name = sensor_record['channel']
        if 'CAM' in sensor_name:
            n_sweeps = self.cfg.CAMERA_SWEEPS
        elif 'RADAR' in sensor_record['channel']:
            n_sweeps = self.cfg.RADAR_SWEEPS
        elif 'LIDAR' in sensor_record['channel']:
            n_sweeps = self.cfg.LIDAR_SWEEPS
        else:
            raise Exception('This should not happen!')
            
        while ind < n_sweeps:
            if not sensor_record['prev'] == "":
                sweeps.append(sensor_record['prev'])
                sensor_record = self.get('sample_data', sensor_record['prev'])
            else:
                break
        return sweeps
    ##--------------------------------------------------------------------------
    def _get_sensor_data(self, idx) -> dict:
        """
        Returns a frame with sensor data in vehicle or global coordinates
        
        :param idx (int): id of the dataset's split to retrieve
        :return sensor_data (dict): all sensor data for that frame
        """
        frame = copy.deepcopy(self.db['frames'][idx])
        ## Get ego pose data
        lidar_rec = self.get('sample_data', frame['lidar'][0]['token'])
        pose_rec = self.get('ego_pose', lidar_rec['ego_pose_token'])
        frame['ego_pose'] = {'translation': pose_rec['translation'], 
                             'rotation': pose_rec['rotation']}
        
        sample_token = lidar_rec['sample_token']
        sample_rec = self.get('sample', sample_token)
        
        ## Get camera data
        for i, cam in enumerate(frame['camera']):
            image, cs_record, filename = self._get_cam_data(cam['token'])
            frame['camera'][i]['image'] = image
            frame['camera'][i]['cs_record'] = cs_record
            frame['camera'][i]['filename'] = filename

        ## Get LIDAR data
        for i, lidar in enumerate(frame['lidar']):
            lidar_pc, lidar_cs = self._get_pointsensor_data('lidar',
                                                            sample_rec,
                                                            lidar_rec,
                                                            pose_rec, 
                                                            self.cfg.LIDAR_SWEEPS)
            frame['lidar'][i]['pointcloud'] = lidar_pc
            frame['lidar'][i]['cs_record'] = lidar_cs

        ## Get Radar data
        all_radar_pcs = RadarPointCloud(np.zeros((18, 0)))
        for i, radar in enumerate(frame['radar']):
            radar_rec = self.get('sample_data', radar['token'])
            pose_rec = self.get('ego_pose', radar_rec['ego_pose_token'])
            radar_pc, _ = self._get_pointsensor_data('radar',
                                                     sample_rec, 
                                                     radar_rec, 
                                                     pose_rec,
                                                     self.cfg.RADAR_SWEEPS)
            all_radar_pcs.points = np.hstack((all_radar_pcs.points, 
                                              radar_pc.points))
        frame['radar'] = [{}]
        frame['radar'][0]['pointcloud'] = all_radar_pcs
 
        ## Get annotations
        frame["anns"] = self._get_annotations(frame, pose_rec)
        self.logger.debug('Annotation Length: {}'.format(len(frame['anns'])))

        return frame
    ##--------------------------------------------------------------------------
    def _get_annotations(self, frame, pose_rec) -> [Box]:
        """
        Gets the annotations for this sample in the vehicle coordinates
        
        :param frame: the frame returned from the db for this sample
        :param pose_record: ego pose record dictionary from nuscenes
        :return: list of Nuscenes Boxes
        """
        if 'test' in self.cfg.SPLIT:
            return []
        else:
            box_list = []
            ## Get boxes from nuscenes in Global coordinates
            orig_box_list = self.get_boxes(frame['lidar'][0]['token'])
            for box in orig_box_list:
                ## Filter boxes based on their class
                try:
                    box.name = _C.NAMEMAPPING[box.name]
                except KeyError:
                    continue
                
                box.velocity = self.box_velocity(box.token)
                
                ## Global to Vehicle
                if self.cfg.COORDINATES == 'vehicle':
                    box.translate(-np.array(pose_rec['translation']))
                    box.rotate(Quaternion(pose_rec['rotation']).inverse)

                box_list.append(box)
        return box_list
    ##--------------------------------------------------------------------------
    def _get_pointsensor_data(self, 
                             sensor_type,
                             sample_rec, 
                             sample_data, 
                             pose_rec=None, 
                             nsweeps=1):
        """
        Returns the LIDAR pointcloud for this frame in vehicle/global coordniates
        
        :param sample_rec: sample record dictionary from nuscenes
        :param sample_data: sample data dictionary from nuscenes
        :param sensor_type (str): 'radar' or 'lidar'
        :param pose_rec: ego pose record dictionary from nuscenes
        :param nsweeps: number of previous sweeps to include
        :return pc: Point cloud
        """
        data_filepath = os.path.join(self.cfg.DATA_ROOT, sample_data['filename'])
        cs_record = self.get('calibrated_sensor', 
                             sample_data['calibrated_sensor_token'])
        ## Read data from file
        if sensor_type == 'lidar':
            pc, _ = LidarPointCloud.from_file_multisweep(self,
                                                        sample_rec, 
                                                        sample_data['channel'], 
                                                        sample_data['channel'], 
                                                        nsweeps=nsweeps,
                                                        min_distance=self.cfg.PC_MIN_DIST)
        elif sensor_type == 'radar':
            pc, _ = RadarPointCloud.from_file_multisweep(self,
                                                        sample_rec, 
                                                        sample_data['channel'], 
                                                        sample_data['channel'], 
                                                        nsweeps=nsweeps,
                                                        min_distance=self.cfg.PC_MIN_DIST)
        else:
            raise Exception('Sensor type not valid.')

        ## Take to vehicle coordinates
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))
       
        ## Take to global coordinates
        if self.cfg.COORDINATES == 'global':
            pc.rotate(Quaternion(pose_rec['rotation']).rotation_matrix)
            pc.translate(np.array(pose_rec['translation']))
        return pc, cs_record
    ##--------------------------------------------------------------------------
    def _get_cam_data(self, cam_token) -> (np.ndarray, np.ndarray):
        """
        Ge camera images and intrinsics

        :param cam_token: sample data token for this camera
        :return image, intrinsics, path:
        """
        ## Get camera image
        filename = self.get_sample_data_path(cam_token)
        cam_data = self.get('sample_data', cam_token)
        cs_record = self.get('calibrated_sensor', 
                                  cam_data['calibrated_sensor_token'])
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                image_str = f.read()
        else:
            raise Exception('Camera image not found at {}'.format(filename))
        image = np.array(Image.open(io.BytesIO(image_str)))
        return image, cs_record, filename
##------------------------------------------------------------------------------
if __name__ == "__main__":
    nusc_ds = NuscenesDataset(cfg='config/cfg.yml')
    print(nusc_ds[0])