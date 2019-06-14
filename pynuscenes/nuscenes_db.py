from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import Box
import threading
import queue
import pickle
from multiprocessing import RLock, Pool, freeze_support
import multiprocessing
import os
import logging
from .utils import constants

class NuscenesDB(object):
    """
    Token database for the nuscenes dataset
    """

    def __init__(self,
                 root_path,
                 nusc_version,
                 max_cam_sweeps=6,
                 max_lidar_sweeps=10,
                 max_radar_sweeps=6,
                 verbose=False):
        """
        Image database object that holds the sample data tokens for the nuscenes dataset
        :param root_path: location of the nuscenes dataset
        :param nusc_version: the version of the dataset to use ('v1.0-trainval', 'v1.0-test', 'v1.0-mini')
        :param max_cam_sweeps: number of sweep tokens to return for each camera
        :param max_lidar_sweeps: number of sweep tokens to return for lidar
        :param max_radar_sweeps: number of sweep tokens to return for each radar
        :param verbose: show debug information
        """

        ## Set up logger
        self.logger = logging.getLogger('pynuscenes')
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        if verbose:
            ch.setLevel(logging.DEBUG)
        else:
            ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(filename)s:%(lineno)d %(levelname)s:: %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
        assert nusc_version in self.available_vers
        self.short_version = str(nusc_version.split('-')[1])
        self.root_path = root_path
        self.nusc_root = os.path.join(root_path, 'datasets', 'nuscenes')
        self.nusc_version = nusc_version
        os.makedirs(os.path.join(self.root_path, 'database', self.nusc_version), exist_ok=True)
        self.max_cam_sweeps = max_cam_sweeps
        self.max_lidar_sweeps = max_lidar_sweeps
        self.max_radar_sweeps = max_radar_sweeps
        self.db = {}
        self.nusc = NuScenes(version=nusc_version, dataroot=self.nusc_root, verbose=True)
        self.SENSOR_NAMES = [x['channel'] for x in self.nusc.sensor]
        self.id_length = 8

    ##--------------------------------------------------------------------------
    def generate_db(self) -> None:
        """
        Create an image databaser (db) for the NuScnenes dataset and save it
        to a pickle file
        """
        self.logger.info('Creating db for the NuScenes dataset ...')
        self._split_scenes()
        train_nusc_frames, val_nusc_frames, test_nusc_frames = self._get_frames()
        metadata = {"version": self.nusc_version}

        if self.nusc_version == 'v1.0-test':
            self.logger.info('Test sample length: {}'.format(str(len(test_nusc_frames))))
            self.db['test'] = {"frames": test_nusc_frames, "metadata": metadata}
            db_filename = "test_db.pkl"
            self.logger.info('Writing pickle file at {}'.format(db_filename))
            with open(os.path.join(self.root_path, 'database', self.nusc_version \
                ,db_filename), 'wb') as f:
                pickle.dump(self.db['test'], f)

        else:
            self.logger.info('Train sample length: {}, val sample length: {}'.format(str(len(train_nusc_frames)), str(len(val_nusc_frames))))
            self.db['train'] = {"frames": train_nusc_frames, "metadata": metadata}
            db_filename = "train_db.pkl"
            with open(os.path.join(self.root_path, 'database', self.nusc_version, db_filename), 'wb') as f:
                pickle.dump(self.db['train'], f)

            self.db['val'] = {"frames": val_nusc_frames, "metadata": metadata}
            db_filename = "val_db.pkl"
            self.logger.info('Writing pickle file at {}'.format(db_filename))
            with open(os.path.join(self.root_path, 'database', self.nusc_version, db_filename), 'wb') as f:
                pickle.dump(self.db['val'], f)

    ##--------------------------------------------------------------------------
    def _split_scenes(self) -> None:
        """
        Split scenes into train, val and test scenes
        """
        scene_split_names = splits.create_splits_scenes()

        self.train_scenes = []
        self.test_scenes = []
        self.val_scenes = []

        for scene in self.nusc.scene:
            #NOTE: mini train and mini val are subsets of train and val
            if scene['name'] in scene_split_names['train']:
                self.train_scenes.append(scene['token'])
            elif scene['name'] in scene_split_names['val']:
                self.val_scenes.append(scene['token'])
            elif scene['name'] in scene_split_names['test']:
                self.test_scenes.append(scene['token'])
            else:
                raise Exception('scene not in splits...split table may not be complete')

        if self.nusc_version == 'v1.0-test':
            self.logger.info('test: {} scenes'.format(str(len(self.test_scenes))))
        else:
            self.logger.info('train: {} scenes, val: {} scenes'.format(str(len(self.train_scenes)), str(len(self.val_scenes))))

    ##------------------------------------------------------------------------------
    def _get_frames(self) -> list:
        """
        returns (train_nusc_frames, val_nusc_frames) from the nuscenes dataset
        """
        self.sample_id = 0

        self.logger.debug('Generating train frames')
        train_nusc_frames = []
        for scene in tqdm(self.train_scenes, desc="train scenes", position=0):
            train_nusc_frames = train_nusc_frames + self.process_scene_samples(scene)

        self.logger.debug('Generating val frames')
        val_nusc_frames = []
        for scene in tqdm(self.val_scenes, desc="val scenes", position=0):
            val_nusc_frames = val_nusc_frames + self.process_scene_samples(scene)

        self.logger.info('Generating test frames')
        test_nusc_frames = []
        for scene in tqdm(self.test_scenes, desc="test scenes", position=0):
            test_nusc_frames = test_nusc_frames + self.process_scene_samples(scene)

        return train_nusc_frames, val_nusc_frames, test_nusc_frames


    ##--------------------------------------------------------------------------
    def process_scene_samples(self, scene: str) -> list:
        """
            Get sensor data and annotations for all samples in the scene.
            :param scene: scene token
            return samples: a list of dictionaries
            frame (a dictionary with a sample, sweeps)
        """
        scene_rec = self.nusc.get('scene', scene)
        scene_number = scene_rec['name'][-4:]
        self.logger.debug('Processing scene {}'.format(scene_number))

        ## Create progress bar
        num_samples = scene_rec['nbr_samples']
        tqdm_position = 1
        ## Get the first sample token in the scene
        sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
        sample_sensor_records = {x: self.nusc.get('sample_data',
            sample_rec['data'][x]) for x in self.SENSOR_NAMES}

        ## Loop over all samples in the scene
        returnList = []
        has_more_samples = True
        while has_more_samples:
            sample = {}
            # sample = {
            #           CAM_FRONT_LEFT: token
            #           CAM_FRONT_RIGHT: token
            #           ...
            #           RADAR_BACK_RIGHT: token
            #          }
            sample.update({cam: sample_sensor_records[cam]['token'] for cam in constants.CAMERAS.keys()})
            sample.update({'LIDAR_TOP': sample_sensor_records['LIDAR_TOP']['token']})
            sample.update({x: sample_sensor_records[x]['token'] for x in constants.RADARS.keys()})

            frame = {'sample': sample,
                     'sweeps': self._get_sweeps(sample_sensor_records),
                     "id": str(self.sample_id).zfill(self.id_length)}
            self.sample_id += 1

            ## Get the next sample if it exists
            if sample_rec['next'] == "":
                has_more_samples = False
            else:
                sample_rec = self.nusc.get('sample', sample_rec['next'])
                sample_sensor_records = {x: self.nusc.get('sample_data',
                    sample_rec['data'][x]) for x in self.SENSOR_NAMES}
            returnList.append(sample)
        return returnList

    ##------------------------------------------------------------------------------
    def _get_sweeps(self, sweep_sensor_records) -> dict:
        """
            :param sweep_sensor_records: list of sample data records for the sensors to return sweeps for
            returns dictionary of lists
                key is the sensor name
                value is the list sweep tokens
        """
        sweep = {x: '' for x in self.SENSOR_NAMES}
        lidar_sweeps = self._get_previous_sensor_sweeps(sweep_sensor_records['LIDAR_TOP'], self.max_lidar_sweeps)

        ## if the LIDAR has no previous sweeps, we assume this is the first sample
        if lidar_sweeps == []:
            return {}

        sweep.update({'LIDAR_TOP': lidar_sweeps})

        for cam in constants.CAMERAS.keys():
            cam_sweeps = {cam: self._get_previous_sensor_sweeps(sweep_sensor_records[cam], self.max_cam_sweeps)}
            sweep.update({cam: cam_sweeps[cam]})

        for radar in constants.RADARS.keys():
            radar_sweeps = {radar: self._get_previous_sensor_sweeps(sweep_sensor_records[radar], self.max_radar_sweeps)}
            sweep.update({radar: radar_sweeps[radar]})

        return sweep

    ##------------------------------------------------------------------------------
    def _get_previous_sensor_sweeps(self, sample_data, num_sweeps) -> list: 
        """
            Gets the previous sweeps for one senser
            :param sample_data: sample_data dictionary for the sensor
            :param num_sweeps: number of sweeps to return
        """
        sweeps = []
        while len(sweeps) < num_sweeps:
            if not sample_data['prev'] == "":
                sweeps.append(sample_data['prev'])
                sample_data = self.nusc.get('sample_data', sample_data['prev'])
            else:
                break
        return sweeps

################################################################################



if __name__ == "__main__":
    test_db()
