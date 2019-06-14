from nuscenes_imdb import NuscenesIMDB
import os
import pickle
def test_imdb():
    nuscenes_path = '/home/cavs/datasets/nuscenes'
    nuscenes_version = "v1.0-test"

    imdb = NuscenesIMDB(nuscenes_path, nusc_version=nuscenes_version, verbose=True)
    imdb.generate_imdb()
    train_samples = val_samples = test_samples = 0

    if nuscenes_version in ['v1.0-trainval','v1.0-mini']:
        with open(os.path.join(nuscenes_path, "%s_imdb_train.pkl" % str(imdb.short_version)), 'rb') as f:
            data = pickle.load(f)
            train_samples = len(data['frames'])
        print('number of train samples', train_samples)
        
        with open(os.path.join(nuscenes_path, "%s_imdb_val.pkl" % str(imdb.short_version)), 'rb') as f:
            data = pickle.load(f)
            val_samples = len(data['frames'])
        print('number of val samples', val_samples)
    else:
        with open(os.path.join(nuscenes_path, "%s_imdb_test.pkl" % str(imdb.short_version)), 'rb') as f:
            data = pickle.load(f)
            test_samples = len(data['frames'])
        print('number of test samples', test_samples)
        print(len(imdb.nusc.sample))
    
    assert len(imdb.nusc.sample) == train_samples + val_samples + test_samples, \
        'Length of nuscenes samples does not match number of samples in imdb'
    
    print('Passed!')

if __name__ == "__main__":
    test_imdb()