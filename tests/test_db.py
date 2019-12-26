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
    logger = logging.getLogger(__name__)
    root = FLAGS.data_location
    passed = True
    for nuscenes_version in ['v1.0-mini']:
        num_samples = 0
        for split in constants.NUSCENES_SPLITS[nuscenes_version]:
            nuscenes_db = NuscenesDB(nusc_root=root, 
                                     nusc_version=nuscenes_version, 
                                     split=split)
            nuscenes_db.generate_db()
            num_samples += len(nuscenes_db.db['frames'])

        if len(nuscenes_db.nusc.sample) != num_samples:
            logger.critical('Length of nuscenes samples does not match samples \
                             in db for {}'.format(nuscenes_version))
            logger.critical('length should be {}, but got {}'.format(len( \
                            nuscenes_db.nusc.sample), num_samples))
            passed = False
        else:
            passed = True
    if passed:    
        logger.info('Tests Passed!')
    else:
        logger.error('Test failed...see output above')

if __name__ == "__main__":
    test_nuscenes_db()