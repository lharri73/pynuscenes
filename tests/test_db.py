from context import pynuscenes
import os
import pickle
import logging
def test_nuscenes_db():
    logger = logging.getLogger('pynuscenes')
    root = '/CAVS/object_detection/nuscenes_dataset/data'
    nuscenes_version = "v1.0-test"
    database_path = os.path.join(root, 'database', nuscenes_version)

    nuscenes_db = pynuscenes.NuscenesDB(root, nusc_version=nuscenes_version)
    nuscenes_db.generate_db()
    train_samples = val_samples = test_samples = 0

    if nuscenes_version in ['v1.0-trainval','v1.0-mini']:
        with open(os.path.join(database_path, "train_db.pkl"), 'rb') as f:
            data = pickle.load(f)
            train_samples = len(data['frames'])
        logger.info('{} train samples'.format(train_samples))
        
        with open(os.path.join(database_path, "val_db.pkl"), 'rb') as f:
            data = pickle.load(f)
            val_samples = len(data['frames'])
        logger.info('{} val samples'.format(val_samples))
    else:
        with open(os.path.join(database_path, "test_db.pkl"), 'rb') as f:
            data = pickle.load(f)
            test_samples = len(data['frames'])
        logger.info('{} test samples'.format(test_samples))
    
    assert len(nuscenes_db.nusc.sample) == train_samples + val_samples + test_samples, \
        'Length of nuscenes samples does not match number of samples in nuscenes_db'
    
    logger.info('Passed with {} total samples!'.format(len(nuscenes_db.nusc.sample)))

if __name__ == "__main__":
    test_nuscenes_db()