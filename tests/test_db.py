################################################################################
## Date Created  : Fri Jun 14 2019                                            ##
## Authors       : Landon Harris, Ramin Nabati                                ##
## Last Modified : Sat Jun 15 2019                                            ##
## Copyright (c) 2019                                                         ##
################################################################################

import context 
from pynuscenes.utils import constants
from pynuscenes.nuscenes_db import NuscenesDB
import os
import pickle
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_location', type=str, default='../data/nuscenes')
parser.add_argument('--versions')

FLAGS = parser.parse_args()

def test_nuscenes_db():
    logger = logging.getLogger('pynuscenes')
    root = FLAGS.data_location
    nusc = None
    passed = True
    for nuscenes_version in ['v1.0-mini']:
        num_samples = 0
        for split in constants.NUSCENES_SPLITS[nuscenes_version]:
            nuscenes_db = NuscenesDB(root, nusc_version=nuscenes_version, split=split, nusc=nusc)
            nuscenes_db.generate_db()
            num_samples += len(nuscenes_db.db['frames'])

        if len(nuscenes_db.nusc.sample) != num_samples:
            logger.critical('Length of nuscenes samples does not match samples in db for {}'.format(nuscenes_version))
            logger.critical('length should be {}, but got {}'.format(len(nuscenes_db.nusc.sample), num_samples))
            passed = False
        else:
            passed = True
    if passed:    
        logger.info('Passed!')
    else:
        logger.error('Test failed...see output above')
    logger.info('Waiting to remove nuscenes from memory...')

if __name__ == "__main__":
    test_nuscenes_db()