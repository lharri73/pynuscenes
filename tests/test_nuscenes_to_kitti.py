import unittest
from pynuscenes.tools.nuscenes_to_kitti import KittiConverter

class TestNuscToKitti(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        converter = KittiConverter(nusc_kitti_dir='./nusc_kitti',
                                    cam_name='CAM_FRONT',
                                    lidar_name='LIDAR_TOP',
                                    lidar_sweeps=10,
                                    radar_sweeps=1,
                                    image_count=10,
                                    nusc_version='v1.0-mini',
                                    split='mini_train')
        
    
    def test_camera(self):
        self.assertEqual('1', '1')
