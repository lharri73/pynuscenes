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
from nuscenes.utils.data_classes import Box, LidarPointCloud, RadarPointCloud, PointCloud

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
        assert self.cfg.POINTCLOUD_MODE in ['camera', 'sample'], \
            'Pointcloud mode not valid'

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
        
        if self.cfg.POINTCLOUD_MODE == 'sample':
            return self.get_sensor_data_by_sample(idx)
        elif self.cfg.POINTCLOUD_MODE == 'camera':
            return self.get_sensor_data_by_camera(idx)
    ##--------------------------------------------------------------------------
    def __len__(self):
        """
        Get the number of samples in the dataset

        return len (int): number of sample in the dataset
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

        # if self.cfg.ENABLE_SWEEPS:
        #     sweeps[sensor_name] = self._get_sweep_tokens(sensor_rec)
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
    def get_sensor_data_by_camera(self, idx) -> dict:
        """
        Returns sensor data in vehicle or global coordinates filtered for each
        camera

        :param idx: id of the dataset's split to retrieve
        :return ret_frame: dictionary containing all sensor data for that frame
        """
        frame = self.get_sensor_data_by_sample(idx)
        ret_frame = {
            'camera': frame['camera'],
            'radar': [],
            'lidar': [],
            'annotations': [],
            'ego_pose': frame['ego_pose'],
            'img_id': [],
            'id': frame['id']
        }
        for i, cam in enumerate(frame['camera']): 
            ret_frame['img_id'].append(str(idx*6+i).zfill(self.cfg.IMG_ID_LEN))

            if 'lidar' in self.sensors_to_return:
                lidar_pc = self.filter_points(frame['lidar']['points'].points, 
                                              cam['cs_record'])
                ret_frame['lidar'].append(lidar_pc)

            if 'radar' in self.sensors_to_return:
                radar_pc = self.filter_points(frame['radar']['points'].points, 
                                              cam['cs_record'])
                ret_frame['radar'].append(radar_pc)

            annotation = self.filter_anns(frame['annotations'], cam['cs_record'],
                                          img=cam['image'])
            
            ret_frame['annotations'].append(annotation)
        return ret_frame
    ##--------------------------------------------------------------------------
    def get_sensor_data_by_sample(self, idx) -> dict:
        """
        Returns sensor data in vehicle or global coordinates
        
        :param idx (int): id of the dataset's split to retrieve
        :return sensor_data (dict): all sensor data for that frame
        """
        frame = copy.deepcopy(self.db['frames'][idx])
        ## Get ego pose data
        lidar_rec = self.get('sample_data', frame['lidar'][0]['token'])
        ego_pose_token = lidar_rec['ego_pose_token']
        pose_rec = self.get('ego_pose', ego_pose_token)
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

        ## TODO: return numpy arrays for pointclouds to match get_sensor_data_by_sample
        ## Get LIDAR data
        for i, lidar in enumerate(frame['lidar']):
            lidar_pc, lidar_cs_record = self._get_lidar_data(sample_rec, 
                                                lidar_rec, pose_rec, 
                                                self.cfg.LIDAR_SWEEPS)
            frame['lidar'][i]['pointcloud'] = lidar_pc
            frame['lidar'][i]['cs_record'] = lidar_cs_record

        ## Get Radar data
        all_radar_pcs = RadarPointCloud(np.zeros((18, 0)))
        for i, radar in enumerate(frame['radar']):
            sample_data = self.get('sample_data', radar['token'])
            current_radar_pc = self._get_radar_data(sample_rec, 
                                                    sample_data, 
                                                    self.cfg.RADAR_SWEEPS)
            ## Vehicle to global coordinate
            if self.cfg.COORDINATES == 'global':
                current_radar_pc.rotate(Quaternion(pose_rec['rotation']).rotation_matrix)
                current_radar_pc.translate(np.array(pose_rec['translation']))

            all_radar_pcs.points = np.hstack((all_radar_pcs.points, 
                                              current_radar_pc.points))
        frame['radar'] = [{}]
        frame['radar'][0]['pointcloud'] = all_radar_pcs
 
        ## Get annotations
        frame["anns"] = self._get_annotations(frame, pose_rec)
        # print('nuscenes dataset', res['lidar']['points'].points.shape)
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
    def _get_radar_data(self, sample_rec, sample_data, 
                        nsweeps) -> RadarPointCloud:
        """
        Returns Radar point cloud in Vehicle Coordinates
        
        :param sample_rec: sample record dictionary from nuscenes
        :param sample_data: sample data dictionary from nuscenes
        :param nsweeps: number of sweeps to return for this radar
        :return pc: RadarPointCloud containing this samnple and all sweeps
        """
        radar_path = os.path.join(self.cfg.DATA_ROOT, sample_data['filename'])
        cs_record = self.get('calibrated_sensor', 
                                  sample_data['calibrated_sensor_token'])
        if nsweeps > 1:
            ## Returns in vehicle coordinates
            pc, _ = RadarPointCloud.from_file_multisweep(self,
                                            sample_rec, 
                                            sample_data['channel'], 
                                            sample_data['channel'], 
                                            nsweeps=nsweeps,
                                            min_distance=self.cfg.RADAR_MIN_DIST)
        else:
            ## Returns in sensor coordinates
            pc = RadarPointCloud.from_file(radar_path)
        ## Sensor to vehicle
        rot_matrix = Quaternion(cs_record['rotation']).rotation_matrix
        pc.rotate(rot_matrix)
        pc.translate(np.array(cs_record['translation']))
        return pc
    ##--------------------------------------------------------------------------
    def _get_lidar_data(self, sample_rec, sample_data, 
                        pose_rec, nsweeps=1) -> LidarPointCloud:
        """
        Returns the LIDAR pointcloud for this frame in vehicle/global coordniates
        
        :param sample_rec: sample record dictionary from nuscenes
        :param sample_data: sample data dictionary from nuscenes
        :param pose_rec: ego pose record dictionary from nuscenes
        :param nsweeps: number of sweeps to return for the LIDAR
        :return: LidarPointCloud containing this sample and all sweeps
        """
        lidar_path = os.path.join(self.cfg.DATA_ROOT, sample_data['filename'])
        cs_record = self.get('calibrated_sensor', 
                                  sample_data['calibrated_sensor_token'])
        if nsweeps > 1:
            ## Returns in vehicle
            lidar_pc, _ = LidarPointCloud.from_file_multisweep(self,
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
        if self.cfg.COORDINATES == 'global':
            lidar_pc.rotate(Quaternion(pose_rec['rotation']).rotation_matrix)
            lidar_pc.translate(np.array(pose_rec['translation']))
        return lidar_pc, cs_record
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
    ##--------------------------------------------------------------------------
    @staticmethod
    def pc_to_sensor(pc_orig, cs_record, coordinates='vehicle', ego_pose=None):
        """
        Transform the input point cloud from global/vehicle coordinates to
        sensor coordinates

        :param pc_orig
        :param cs_record
        :param coordinates (string)
        :ego_pose (dict)
    
        """
        if coordinates == 'global':
            assert ego_pose is not None, \
                'ego_pose is required in global coordinates'
        
        ## Copy is required to prevent the original pointcloud from being manipulate
        pc = copy.deepcopy(pc_orig)
        
        if isinstance(pc, PointCloud):
            if coordinates == 'global':
                ## Transform from global to vehicle
                pc.translate(np.array(-np.array(ego_pose['translation'])))
                pc.rotate(Quaternion(ego_pose['rotation']).rotation_matrix.T)

            ## Transform from vehicle to sensor
            pc.translate(-np.array(cs_record['translation']))
            pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
        elif isinstance(pc, np.ndarray):
            if coordinates == 'global':
                ## Transform from global to vehicle
                for i in range(3):
                    pc[i, :] = pc[i, :] + np.array(-np.array(ego_pose['translation']))[i]
                pc[:3, :] = np.dot(Quaternion(ego_pose['rotation']).rotation_matrix.T, pc[:3, :])
            ## Transform from vehicle to sensor
            for i in range(3):
                pc[i, :] = pc[i, :] - np.array(cs_record['translation'])[i]
            pc[:3, :] = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, pc[:3, :])
        elif isinstance(pc, list):
            if len(pc) == 0:
                return []
            if isinstance(pc[0], Box):
                new_list = []
                for box in pc:
                    if coordinates == 'global':
                        ## Transform from global to vehicle
                        box.translate(-np.array(ego_pose['translation']))
                        box.rotate(Quaternion(ego_pose['rotation']).inverse)

                    ## Transform from vehicle to sensor
                    box.translate(-np.array(cs_record['translation']))
                    box.rotate(Quaternion(cs_record['rotation']).inverse)
                    new_list.append(box)
                return new_list
        elif isinstance(pc, Box):
            if coordinates == 'global':
                ## Transform from global to vehicle
                pc.translate(-np.array(ego_pose['translation']))
                pc.rotate(Quaternion(ego_pose['rotation']).inverse)
            ## Transform from vehicle to sensor
            pc.translate(-np.array(cs_record['translation']))
            pc.rotate(Quaternion(cs_record['rotation']).inverse)
        else:
            raise TypeError('cannot filter object with type {}'.format(type(pc)))

        return pc
    ##--------------------------------------------------------------------------
    @staticmethod
    def filter_points(points_orig, cam_cs_record, img_shape=(1600,900)):
        """
        Filter point cloud to only include the ones mapped inside an image

        :param points: pointcloud or box in the coordinate system of the camera
        :param cam_cs_record: calibrated sensor record of the camera to filter to
        :param img_shape: shape of the image (width, height)
        """
        if isinstance(points_orig, np.ndarray):
            points = NuscenesDataset.pc_to_sensor(points_orig, cam_cs_record)
            viewed_points = view_points(points[:3, :], 
                                    np.array(cam_cs_record['camera_intrinsic']), 
                                    normalize=True)
            visible = np.logical_and(viewed_points[0, :] > 0, 
                                     viewed_points[0, :] < img_shape[0])
            visible = np.logical_and(visible, viewed_points[1, :] < img_shape[1])
            visible = np.logical_and(visible, viewed_points[1, :] > 0)
            visible = np.logical_and(visible, points[2, :] > 1)
            in_front = points[2, :] > 0.1  
            # True if a corner is at least 0.1 meter in front of the camera.
            
            isVisible = np.logical_and(visible, in_front)
            points_orig = points_orig.T[isVisible]
            points_orig = points_orig.T
            return points_orig
        else:
            raise TypeError('{} is not able to be filtered'.format(type(points)))
    ##--------------------------------------------------------------------------
    @staticmethod
    def filter_anns(annotations_orig, cam_cs_record, img_shape=(1600,900), 
                    img=np.zeros((900,1600,3))):
        """
        Filter annotations to only include the ones mapped inside an image

        :param annotations_orig: annotation boxes
        :param cam_cs_record: calibrated sensor record of the camera to filter to
        :param img_shape: shape of the image (width, height)
        """
        if len(annotations_orig) == 0:
            return []
        assert isinstance(annotations_orig[0], Box)
    
        annotations = NuscenesDataset.pc_to_sensor(annotations_orig, cam_cs_record)
        visible_boxes = []
        for i, box in enumerate(annotations):
            if box_in_image(box, np.array(cam_cs_record['camera_intrinsic']), 
                            img_shape):
                # box.render_cv2(img, view=np.array(cam_cs_record['camera_intrinsic']), normalize=True)
                # cv2.imshow('image', img)
                # cv2.waitKey(1)
                visible_boxes.append(annotations_orig[i])
        return visible_boxes
##------------------------------------------------------------------------------
if __name__ == "__main__":
    nusc_ds = NuscenesDataset(cfg='config/cfg.yml')
    print(nusc_ds[0])