import os
import numpy as np
import argparse
from tqdm import trange
import matplotlib.pyplot as plt
from cocoplus.coco import COCO_PLUS
from cocoplus.utils.coco_utils import COCO_CATEGORIES
import pynuscenes.utils.nuscenes_utils as nsutils
# from pynuscenes.utils.nuscenes_utils import nuscenes_box_to_coco, nuscene_cat_to_coco
from pynuscenes.nuscenes_dataset import NuscenesDataset
from nuscenes.nuscenes import NuScenes
from pynuscenes.utils import io_utils
"""
This script converts NuScenes data to COCO format.
"""

class CocoConverter:
    def __init__(self,
                 nusc_root: str = '../../data/nuscenes',
                 nusc_cfg: str = '../config/cfg.yml',
                 output_dir: str = '../../data/nucoco',
                 use_symlinks: bool = False,
                ):
        self.cfg = io_utils.yaml_load(nusc_cfg, safe_load=True)
        self.use_symlinks = use_symlinks

        ## Sanity check
        assert self.cfg.SAMPLE_MODE == 'one_cam', \
            'SAMPLE_MODE must be one_cam for CocoConverter'

        ## Create nuscenes and coco dataset instances
        self.nusc_dataset = NuscenesDataset(nusc_root, cfg=nusc_cfg)    
        self.coco_dataset = COCO_PLUS(logging_level="INFO")
        self.coco_dataset.create_new_dataset(dataset_dir=output_dir,
                                             split=self.cfg.SPLIT)
        ## Set COCO categories
        self.CATEGORIES = [{'supercategory': 'person', 'id': 1, 'name': 'person'}, 
                      {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}, 
                      {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
                      {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},  
                      {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
                      {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'}]
        self.coco_dataset.setCategories(self.CATEGORIES)
    ##--------------------------------------------------------------------------
    def convert(self):
        ## Get samples from the Nuscenes dataset

        for sample in self.nusc_dataset:
            cam = sample['camera'][0]
            image = cam['image']
            cam_cs_rec = cam['cs_record']
            cam_pose_rec = cam['pose_record']
            img_height, img_width, _ = image.shape
            anns = sample['anns']

            ## Create annotation in coco_dataset format
            this_sample_anns = []        
            for ann in anns:
                ## Get equivalent coco category name
                coco_cat, coco_cat_id, coco_supercat = nsutils.nuscene_cat_to_coco(ann.name)
                if coco_cat is None:
                    continue
                
                ## Take annotations to the camera frame
                if self.cfg.COORDINATES == 'global':
                    ann = nsutils.global_to_vehicle(ann, cam_pose_rec)
                ann = nsutils.vehicle_to_sensor(ann, cam_cs_rec)
                
                ## Get 2D bbox from the 3D annotation
                view = np.array(cam_cs_rec['camera_intrinsic'])
                bbox = nsutils.nuscenes_box_to_coco(ann, view, (img_width, img_height))
                if bbox is None:
                    continue
                coco_ann = self.coco_dataset.createAnn(bbox, coco_cat_id)
                this_sample_anns.append(coco_ann)

            ## Get the pointclouds added to dataset
            pc = sample['radar'][0]['pointcloud'].points
            pc_coco = np.transpose(pc).tolist()

            ## Add sample to the COCO dataset
            img_id = cam['img_id']
            coco_img_path = self.coco_dataset.addSample(img=image,
                                        anns=this_sample_anns, 
                                        pointcloud=pc_coco,
                                        img_id=img_id,
                                        other=cam_cs_rec,
                                        img_format='RGB',
                                        write_img= not self.use_symlinks)            
            if self.use_symlinks:
                os.symlink(os.path.abspath(cam['cam_path']), coco_img_path)
            
            # Uncomment to visualize
            ax = self.coco_dataset.showImgAnn(np.asarray(image), this_sample_anns, bbox_only=True, BGR=False)
            plt.show()
            input('here')

        self.coco_dataset.saveAnnsToDisk()
## -----------------------------------------------------------------------------
def parse_args():
    """
    Parser for input arguments
    """
    parser = argparse.ArgumentParser(description='Converts the NuScenes dataset to COCO format')
    
    parser.add_argument('--nusc_cfg', default='../config/cfg.yml',
                        help='NuScenes config file')
    
    parser.add_argument('-o', '--out_dir', default='../../data/nucoco',
                        help='Output directory for the nucoco dataset')
    
    parser.add_argument('-s', '--use_symlinks', action='store_true',
                        dest='use_symlinks',
                        help='Use symlinks to images rather than duplicating them.')
    
    args = parser.parse_args()
    return args
## -----------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    cfg = io_utils.yaml_load(args.nusc_cfg, safe_load=True)
    out_dir = os.path.join(args.out_dir, cfg.VERSION)
    
    converter = CocoConverter(
                nusc_cfg = args.nusc_cfg,
                output_dir = args.out_dir,
                use_symlinks = args.use_symlinks)
    converter.convert()