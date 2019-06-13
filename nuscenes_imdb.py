from tqdm import tqdm
from pathlib import Path
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

class NuscenesIMDB(object):
    """
    Image database for the Nuscenes dataset.
    """
    CAMS = ['CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
            'CAM_BACK']

    CAM_ID = { 'CAM_FRONT_LEFT':  0,
               'CAM_FRONT_RIGHT': 1,
               'CAM_FRONT':       2,
               'CAM_BACK_LEFT':   3,
               'CAM_BACK_RIGHT':  4,
               'CAM_BACK':        5}

    RADARS = ['RADAR_FRONT_LEFT',
              'RADAR_FRONT_RIGHT',
              'RADAR_FRONT',
              'RADAR_BACK_LEFT',
              'RADAR_BACK_RIGHT']

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
        self.logger = logging.getLogger('imdb_generator')
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        if verbose:
            ch.setLevel(logging.DEBUG)
        else:
            ch.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(pathname)s:%(lineno)d %(levelname)s:: %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
        assert nusc_version in self.available_vers
        self.short_version = str(nusc_version.split('-')[1])
        self.root_path = root_path
        self.nusc_version = nusc_version
        self.max_cam_sweeps = max_cam_sweeps
        self.max_lidar_sweeps = max_lidar_sweeps
        self.max_radar_sweeps = max_radar_sweeps
        self.is_test = "test" in self.nusc_version
        self.imdb = {}
        self.nusc = NuScenes(version=nusc_version, dataroot=root_path, verbose=True)
        self.SENSOR_NAMES = [x['channel'] for x in self.nusc.sensor]
        self.id_length = 8

    ##--------------------------------------------------------------------------
    def generate_imdb(self):
        """
        Create an image databaser (imdb) for the NuScnenes dataset and save it
        to a pickle file
        """
        self.logger.info('creating imdb for the NuScenes dataset ...')
        self._split_scenes()
        train_nusc_frames, val_nusc_frames, test_nusc_frames = self._get_frames()
        metadata = {"version": self.nusc_version}

        if self.is_test:
            self.logger.info('test sample length: {}'.format(str(len(test_nusc_frames))))
            self.imdb['test'] = {"frames": test_nusc_frames, "metadata": metadata}
            imdb_filename = "{}_imdb_test.pkl".format(self.short_version)
            with open(os.path.join(self.root_path, imdb_filename), 'wb') as f:
                pickle.dump(self.imdb['test'], f)

        else:
            self.logger.info('train sample length: {}, val sample length: {}'.format(str(len(train_nusc_frames)), str(len(val_nusc_frames))))
            self.imdb['train'] = {"frames": train_nusc_frames, "metadata": metadata}
            imdb_filename = "{}_imdb_train.pkl".format(self.short_version)
            with open(os.path.join(self.root_path, imdb_filename), 'wb') as f:
                pickle.dump(self.imdb['train'], f)

            self.imdb['val'] = {"frames": val_nusc_frames, "metadata": metadata}
            imdb_filename = "{}_imdb_val.pkl".format(self.short_version)
            with open(os.path.join(self.root_path, imdb_filename), 'wb') as f:
                pickle.dump(self.imdb['val'], f)


    ##--------------------------------------------------------------------------
    def _split_scenes(self):
        """
        Split scenes into train, val and test scenes
        """
        scene_split_mapping = splits.create_splits_scenes()

        if self.nusc_version == "v1.0-trainval" or self.nusc_version == "trainval":
            train_scenes = scene_split_mapping['train']
            val_scenes = scene_split_mapping['val']
            test_scenes = []

        elif self.nusc_version == "v1.0-test" or self.nusc_version == "test":
            train_scenes = []
            val_scenes = []
            test_scenes = scene_split_mapping['test']

        elif self.nusc_version == "v1.0-mini" or self.nusc_version == "mini":
            train_scenes = scene_split_mapping['mini_train']
            val_scenes = scene_split_mapping['mini_val']
            test_scenes = []

        else:
            raise ValueError("Unknown NuScenes version " + self.nusc_version)
        if self.is_test:
            self.logger.info('test: {} scenes'.format(str(len(test_scenes))))
        else:
            self.logger.info('train: {} scenes, val: {} scenes'.format(str(len(train_scenes)), str(len(val_scenes))))

        self.train_scenes = train_scenes
        self.val_scenes = val_scenes
        self.test_scenes = test_scenes

    ##------------------------------------------------------------------------------
    def _get_frames(self):
        """
        returns (train_nusc_frames, val_nusc_frames) from the nuscenes dataset
        """
        self.sample_id = 0

        train_nusc_frames = []
        for scene in tqdm(self.train_scenes, desc="train scenes", position=0):
            train_nusc_frames = train_nusc_frames + self.process_scene_samples(scene)

        val_nusc_frames = []
        for scene in tqdm(self.val_scenes, desc="val scenes", position=0):
            val_nusc_frames = val_nusc_frames + self.process_scene_samples(scene)

        test_nusc_frames = []
        for scene in tqdm(self.test_scenes, desc="test scenes", position=0):
            test_nusc_frames = test_nusc_frames + self.process_scene_samples(scene)

        return train_nusc_frames, val_nusc_frames, test_nusc_frames


    ##--------------------------------------------------------------------------
    def process_scene_samples(self, scene) -> [{}]:
        """
            Get sensor data and annotations for all samples in the scene.
            Returns a list of dictionaries
            frame (a dictionary with a sample, sweeps)
        """


        scene_number = scene['name'][-4:]
        scene_token = scene["token"]
        scene_rec = self.nusc.get('scene', scene_token)

        ## Create progress bar
        num_samples = scene['nbr_samples']
        tqdm_position = 1
        pbar = tqdm(total=num_samples, desc='scene {}'.format(scene_number), position=(tqdm_position))

        ## Get the first sample token in the scene
        sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
        sample_sensor_records = {x: self.nusc.get('sample_data',
            sample_rec['data'][x]) for x in self.SENSOR_NAMES}

        ## Loop over all samples in the scene
        returnList = []
        has_more_samples = True
        while has_more_samples:
            sample = {}
            #   sample = {
            #               CAM_FRONT_LEFT: token
            #               CAM_FRONT_RIGHT: token
            #               ...
            #               RADAR_BACK_RIGHT: token
            #             }
            sample.update({cam: sample_sensor_records[cam]['token'] for cam in self.CAMS})
            sample.update({'LIDAR_TOP': sample_sensor_records['LIDAR_TOP']['token']})
            sample.update({x: sample_sensor_records[x]['token'] for x in self.RADARS})

            frame = {'sample': sample,
                     'sweeps': self._get_sweeps(sample_sensor_records),
                     "id": str(self.sample_id).zfill(self.id_length)}
            self.sample_id += 1
            pbar.update(1)

            ## Get the next sample if it exists
            if sample_rec['next'] == "":
                has_more_samples = False
            else:
                sample_rec = self.nusc.get('sample', sample_rec['next'])
                sample_sensor_records = {x: self.nusc.get('sample_data',
                    sample_rec['data'][x]) for x in self.SENSOR_NAMES}

        pbar.close()
        return returnList


    ##------------------------------------------------------------------------------
    def _get_previous_sensor_sweeps(self, sample_data, num_sweeps) -> [str]: # list of tokens
        sweeps = []
        while len(sweeps) < num_sweeps:
            if not sample_data['prev'] == "":
                sweeps.append(sample_data['prev'])
                sample_data = self.nusc.get('sample_data', sample_data['prev'])
            else:
                break
        return sweeps


    ##------------------------------------------------------------------------------
    def _get_sweeps(self, sweep_sensor_records):
        """
            returns dictionary of lists
        """
        sweep = {x: '' for x in self.SENSOR_NAMES}
        lidar_sweeps = self._get_previous_sensor_sweeps(sweep_sensor_records['LIDAR_TOP'], self.max_lidar_sweeps)

        ## if the LIDAR has no previous sweeps, we assume this is the first sample
        if lidar_sweeps == []:
            return {}

        sweep.update({'LIDAR_TOP': lidar_sweeps})

        for cam in self.CAMS:
            cam_sweeps = {cam: self._get_previous_sensor_sweeps(sweep_sensor_records[cam], self.max_cam_sweeps)}
            sweep.update({cam: cam_sweeps[cam]})

        for radar in self.RADARS:
            radar_sweeps = {radar: self._get_previous_sensor_sweeps(sweep_sensor_records[radar], self.max_radar_sweeps)}
            sweep.update({radar: radar_sweeps[radar]})

        return sweep


    ##------------------------------------------------------------------------------
    def _get_available_scenes(self):
        available_scenes = []

        for scene in self.nusc.scene:
            scene_token = scene["token"]
            scene_rec = self.nusc.get('scene', scene_token)
            sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = self.nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])
            has_more_frames = True
            scene_not_exist = False
            while has_more_frames:
                lidar_path, boxes, _ = self.nusc.get_sample_data(sd_rec['token'])
                if not Path(lidar_path).exists():
                    scene_not_exist = True
                    break
                else:
                    break

                ##FIXME: Do we ever get to this point?
                if not sd_rec['next'] == "":
                    sd_rec = self.nusc.get('sample_data', sd_rec['next'])
                else:
                    has_more_frames = False
            if scene_not_exist:
                continue
            available_scenes.append(scene)
        print("existing scenes number:", len(available_scenes))
        return available_scenes

    ##--------------------------------------------------------------------------
    def process_scene_sweeps(self, scene):
        """
            Get sensor data and annotations for all sweeps in the scene. Sweeps
            are data from sensors with no annotation. The annotation is calculated
            using interpolation from closest samples.
            Returns a list of dictionaries, the key is if its a train or val, and the key is a
            frame (a dictionary with a sample, sweeps, and whether it's a test scene or not)
        """
        ##TODO: Test this function then uncomment and remvove NotImplementedError
        raise NotImplementedError("unable to create imdb with sweeps")

        # train_nusc_frames = []
        # val_nusc_frames = []
        # returnList = []
        # counter = scene['start_counter']

        # scene_number = scene['name'][-4:]
        # scene_token = scene["token"]
        # scene_rec = self.nusc.get('scene', scene_token)
        # sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])

        # ## Create progress bar
        # num_sweeps = round((scene['nbr_samples'])* 5.6 * len(self.CAMS_TO_RADARS.keys()))
        # tqdm_position = multiprocessing.current_process()._identity[0]
        # pbar = tqdm(total=num_sweeps, desc='scene {}'.format(scene_number), position=(tqdm_position))

        # ## Loop over all cameras in the scene
        # for cam in self.CAMS_TO_RADARS.keys():
        #     sweep_sensor_records = {x: self.nusc.get('sample_data',
        #         sample_rec['data'][x]) for x in self.SENSOR_NAMES}
        #     sweep_sensor_tokens = {x: sweep_sensor_records[x]['token']
        #         for x in self.SENSOR_NAMES}
        #     has_more_sweeps = True

        #     ## Get all sensor sweeps
        #     while has_more_sweeps:
        #         counter += 1

        #         lidar_path, boxes, _ = self.nusc.get_sample_data(
        #             sweep_sensor_records['LIDAR_TOP']['token'])
        #         CAM_sample_data = sweep_sensor_records[cam]
        #         cam_path, _, cam_intrinsic = self.nusc.get_sample_data(
        #             CAM_sample_data['token'])

        #         radar_paths = []
        #         for radar in self.CAMS_TO_RADARS[cam]:
        #             RADAR_sample_data = sweep_sensor_records[radar]
        #             radar_path = self.nusc.get_sample_data_path(RADAR_sample_data['token'])
        #             radar_paths.append(radar_path)

        #         _id = self.start_counter + int(scene_number)*(10**5) + self.CAM_ID[cam]*(10**4) + counter

        #         sample = {
        #             'cam_token': sweep_sensor_tokens[cam],
        #             'lidar_token': sweep_sensor_tokens['LIDAR_TOP'],
        #             'radar_tokens': [sweep_sensor_tokens[x] for x in self.CAMS_TO_RADARS[cam]]
        #             }

        #         frame = {
        #                 'sample': sample,
        #                 'sweeps': self._get_sweeps(sweep_sensor_records),
        #                 'is_test': self.is_test,
        #                 "id": _id
        #                 }

        #         if scene_token in self.train_scenes:
        #             returnList.append({'type': 'trainScene', 'frame': frame})
        #         else:
        #             returnList.append({'type': 'valScene', 'frame': frame})
        #         pbar.update(1)

        #         ## Go to the next sample
        #         if sweep_sensor_records[cam]['next'] == "" or sweep_sensor_records['LIDAR_TOP']['next'] == "":
        #             has_more_sweeps = False
        #         else:
        #             for sensor in self.SENSOR_NAMES:
        #                 if not sweep_sensor_records[sensor] == {} and not sweep_sensor_records[sensor]['next'] == "":
        #                     next_token = sweep_sensor_records[sensor]['next']
        #                     sweep_sensor_tokens[sensor] = next_token
        #                     sweep_sensor_records[sensor]= self.nusc.get('sample_data', next_token)
        #                 else:
        #                     sweep_sensor_records[sensor] = {}

        # pbar.close()
        # return returnList


################################################################################
def test_imdb():
    nuscenes_path = '/home/cavs/wad-challenge/data/datasets/nuscenes'
    nuscenes_version = "v1.0-mini"

    imdb = NuscenesIMDB(nuscenes_path, nusc_version=nuscenes_version)
    with open(os.path.join(nuscenes_path, "%s_imdb_train.pkl" % str(imdb.short_version)), 'rb') as f:
        data = pickle.load(f)


if __name__ == "__main__":
    test_imdb()
