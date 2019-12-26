import os
import numpy as np
import argparse
from tqdm import trange
import matplotlib.pyplot as plt
from cocoplus.coco import COCO_PLUS
from cocoplus.utils.coco_utils import COCO_CATEGORIES
from pynuscenes.utils.nuscenes_utils import nuscenes_box_to_coco, nuscene_cat_to_coco
from pynuscenes.nuscenes_dataset import NuscenesDataset
from nuscenes.nuscenes import NuScenes

"""
This script converts nuScenes data to COCO format.
"""

class COCOConverter:
    def __init__(self,
                 output_dir: str = '../../data/nucoco',
                 nusc_dir: str = '../../data/nuscenes',
                 nusc_version: str = 'v1.0-mini',
                 split: str = 'mini_train',
                 coordinates: str = 'vehicle',
                 use_symlinks: bool = False,
                 radar_sweeps: int=1,
                 cameras: list = ['CAM_FRONT',
                                 'CAM_BACK',
                                #  'CAM_FRONT_LEFT',
                                #  'CAM_FRONT_RIGHT',
                                #  'CAM_BACK_LEFT',
                                #  'CAM_BACK_RIGHT',
                                 ],
                ):

        self.cameras = cameras
        self.use_symlinks = use_symlinks
        
        ## Create a Nuscenes dataloader instance
        self.nusc_dataset = NuscenesDataset(nusc_path=nusc_dir, 
                                            nusc_version=nusc_version, 
                                            split=split,
                                            coordinates=coordinates,
                                            radar_sweeps=radar_sweeps,
                                            sensors_to_return=['camera', 'radar'],
                                            pc_mode='camera',
                                            logging_level='INFO')
    
        ## Create an empty COCO dataset
        self.coco_dataset = COCO_PLUS(logging_level="INFO")
        self.coco_dataset.create_new_dataset(dataset_dir=output_dir, split=split)

        ## Set COCO categories
        self.CATEGORIES = [{'supercategory': 'person', 'id': 1, 'name': 'person'}, 
                      {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}, 
                      {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
                      {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},  
                      {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
                      {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'}]
        self.coco_dataset.setCategories(self.CATEGORIES)


    def convert(self):
        ## Get samples from the Nuscenes dataset
        num_samples = len(self.nusc_dataset)
        for i in trange(num_samples):
            sample = self.nusc_dataset[i]
            img_ids = sample['img_id']

            for i, cam_sample in enumerate(sample['camera']):
                if cam_sample['camera_name'] not in self.cameras:
                    continue

                img_id = int(img_ids[i])
                image = cam_sample['image']
                pc = sample['radar'][i]
                cam_cs_record = cam_sample['cs_record']
                img_height, img_width, _ = image.shape

                # Create annotation in coco_dataset format
                this_sample_anns = []
                annotations = self.nusc_dataset.pc_to_sensor(sample['annotations'][i], 
                                                        cam_cs_record)
            
                for ann in annotations:
                    coco_cat, coco_cat_id, coco_supercat = nuscene_cat_to_coco(ann.name)
                    if coco_cat is None:
                        continue
                    
                    bbox = nuscenes_box_to_coco(ann, np.array(cam_cs_record['camera_intrinsic']), 
                                                (img_width, img_height))
                    coco_ann = self.coco_dataset.createAnn(bbox, coco_cat_id)
                    this_sample_anns.append(coco_ann)

                ## Get the pointclouds added to dataset
                pc_coco = np.transpose(pc).tolist()

                ## Add sample to the COCO dataset
                coco_img_path = self.coco_dataset.addSample(img=image,
                                            anns=this_sample_anns, 
                                            pointcloud=pc_coco,
                                            img_id=img_id,
                                            other=cam_cs_record,
                                            img_format='RGB',
                                            write_img= not self.use_symlinks,
                                            )            
                if self.use_symlinks:
                    os.symlink(os.path.abspath(cam_sample['cam_path']), coco_img_path)
                
                ## Uncomment to visualize
                # ax = self.coco_dataset.showImgAnn(np.asarray(image), this_sample_anns, bbox_only=True, BGR=False)
                # plt.show()
                # input('here')

        self.coco_dataset.saveAnnsToDisk()


## -----------------------------------------------------------------------------
def parse_args():
    """
    Parse the input arguments
    """
    parser = argparse.ArgumentParser(description='Converts the NuScenes dataset to COCO format')
    
    parser.add_argument('--nusc_root', default='../../data/nuscenes',
                        help='NuScenes dataroot')
    
    parser.add_argument('-v', '--nusc_version', default='v1.0-mini',
                        help="Dataset version ('v1.0-mini', 'v1.0-trainval', 'v1.0-test')")
    
    parser.add_argument('--split', default='mini_val',
                        help='Dataset split (mini_train, mini_val, train, val, test)')
    
    parser.add_argument('-o', '--out_dir', default='../../data/nucoco',
                        help='Output directory for the nucoco dataset')
    
    parser.add_argument('-n', '--radar_sweeps', default=1,
                        help='Number of Radar sweeps to include')
    
    parser.add_argument('-s', '--use_symlinks', action='store_true',
                        dest='use_symlinks',
                        help='Use symlinks to images rather than duplicating them.')
    
    parser.add_argument('-l', '--logging_level', default='INFO',
                        help='Logging level')
    
    parser.add_argument('-c' , '--cameras', nargs='+',
                        default=['CAM_FRONT',
                                 'CAM_BACK',
                                #  'CAM_FRONT_LEFT',
                                #  'CAM_FRONT_RIGHT',
                                #  'CAM_BACK_LEFT',
                                #  'CAM_BACK_RIGHT',
                                 ],
                        help='List of cameras to use.')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    out_dir = os.path.join(args.out_dir, args.nusc_version)
    
    converter = COCOConverter(
                output_dir = args.out_dir,
                nusc_dir = args.nusc_root,
                nusc_version = args.nusc_version,
                split = args.split,
                coordinates = 'vehicle',
                radar_sweeps=args.radar_sweeps,
                use_symlinks = args.use_symlinks)
    converter.convert()