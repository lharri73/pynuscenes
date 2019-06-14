from context import pynuscenes
import os
import pickle
import logging
def test_nuscenes_db():
    logger = logging.getLogger('pynuscenes')
    root = '/CAVS/object_detection/nuscenes_dataset/data/datasets/nuscenes'
    nuscenes_version = "v1.0-test"
    database_path = os.path.join(root, 'database', nuscenes_version)
    for nusc_version in 
    nuscenes_db = pynuscenes.NuscenesDB(root, nusc_version=nuscenes_version)
    nuscenes_db.generate_db()
    train_samples = val_samples = test_samples = 0

    
    assert len(nuscenes_db.nusc.sample) == train_samples + val_samples + test_samples, \
        'Length of nuscenes samples does not match number of samples in nuscenes_db'
    
    logger.info('Passed with {} total samples!'.format(len(nuscenes_db.nusc.sample)))

if __name__ == "__main__":
    test_nuscenes_db()