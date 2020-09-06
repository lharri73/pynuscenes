import os
import numpy as np
import argparse
from tqdm import trange
import matplotlib.pyplot as plt
from cocoplus.coco import COCO_PLUS
from cocoplus.utils.coco_utils import COCO_CATEGORIES
import pynuscenes.utils.nuscenes_utils as nsutils
from pynuscenes.nuscenes_dataset import NuscenesDataset
from nuscenes.nuscenes import NuScenes
from pynuscenes.utils import log, io_utils
"""
This script converts NuScenes data to COCO format.
"""

class CocoConverter:

    ## Mapping from NuScenes categoies to COCO categoies
    NUSC_COCO_CAT_MAP = {
        'car':                  {'id': 1, 'name': 'car', 'supercategory': 'vehicle'},
        'truck':                {'id': 2, 'name': 'truck', 'supercategory': 'vehicle'},
        'trailer':              {'id': 3, 'name': 'trailer', 'supercategory': 'vehicle'},
        'pedestrian':           {'id': 4, 'name': 'pedestrian', 'supercategory': 'person'},
        'bus':                  {'id': 5, 'name': 'bus', 'supercategory': 'vehicle'},
        'bicycle':              {'id': 6, 'name': 'bicycle', 'supercategory': 'vehicle'},
        'motorcycle':           {'id': 7, 'name': 'motorcycle', 'supercategory': 'vehicle'},
        'construction_vehicle': {'id': 8, 'name': 'construction_vehicle', 'supercategory': 'vehicle'},
    }

    def __init__(self,
                 nusc_root: str = '../../data/nuscenes',
                 nusc_cfg: str = '../config/cfg.yml',
                 output_dir: str = '../../data/nucoco',
                 use_symlinks: bool = False,
                ):
        self.cfg = io_utils.yaml_load(nusc_cfg, safe_load=True)
        self.use_symlinks = use_symlinks
        self.logger = log.getLogger(__name__)

        ## Sanity check
        assert self.cfg.SAMPLE_MODE == 'one_cam', \
            'SAMPLE_MODE must be one_cam for CocoConverter'

        ## Create nuscenes and coco dataset instances
        self.nusc_dataset = NuscenesDataset(nusc_root, cfg=nusc_cfg)    
        self.coco_dataset = COCO_PLUS(logging_level="INFO")
        self.coco_dataset.create_new_dataset(dataset_dir=output_dir,
                                             split=self.cfg.SPLIT)
        ## Set COCO categories
        self.CATEGORIES = [vals for _, vals in self.NUSC_COCO_CAT_MAP.items()]
        self.coco_dataset.setCategories(self.CATEGORIES)
    ##--------------------------------------------------------------------------
    def convert(self):
        """
        Start converting nuscenes_dataset samples to COCO format
        """

        self.logger.info('Converting NuScenes samples to COCO format')
        ## Get samples from the Nuscenes dataset
        for ind in trange(len(self.nusc_dataset)):
            sample = self.nusc_dataset[ind]
            cam = sample['camera'][0]
            image = cam['image']
            cam_cs_rec = cam['cs_record']
            cam_pose_rec = cam['pose_record']
            ref_pose_rec = sample['ref_pose_record']
            img_height, img_width, _ = image.shape
            coordinates = sample['coordinates']
            anns = sample['anns']

            ## Create annotation in coco_dataset format
            this_sample_anns = []  
            for ann in anns:
                ## Get equivalent coco category name
                coco_cat, coco_cat_id, coco_supercat = self.nuscene_cat_to_coco(ann.name)
                if coco_cat is None:
                    continue
                
                ## Take annotations to the camera frame
                ann = nsutils.map_annotation_to_camera(ann, cam_cs_rec, cam_pose_rec, 
                                                       ref_pose_rec, coordinates)
                
                ## Get 2D bbox from the 3D annotation
                view = np.array(cam_cs_rec['camera_intrinsic'])
                bbox = nsutils.box_3d_to_2d_simple(ann, view, (img_width, img_height))
                if bbox is None:
                    continue
                
                ## Get distance to the box from camera
                dist = self.get_box_dist_to_cam(ann)

                coco_ann = self.coco_dataset.createAnn(bbox, coco_cat_id, distance=dist)
                this_sample_anns.append(coco_ann)
            
            ## Get the Radar pointclouds added to dataset
            pc_coco = None
            if 'radar' in sample:
                pc = sample['radar']['pointcloud']
                ## Transform to camera coordinates
                pc, _ = nsutils.map_pointcloud_to_camera(pc, cam_cs_rec, cam_pose_rec,
                                                    sample['radar']['pose_record'],
                                                    coordinates=coordinates)
                cost = np.transpose(pc.points).tolist()
                # print(pc_coco)
                # input('here')

            ## Add sample to the COCO dataset
            img_id = cam['image_id']
            coco_img_path = self.coco_dataset.addSample(img=image,
                                        anns=this_sample_anns, 
                                        pointcloud=pc_coco,
                                        img_id=img_id,
                                        other=cam_cs_rec,
                                        img_format='RGB',
                                        write_img= not self.use_symlinks)            
            if self.use_symlinks:
                try:
                    os.symlink(os.path.abspath(cam['filename']), coco_img_path)
                except FileExistsError as e:
                    self.logger.warning("Symlink '{}' already exists, not overwriting.".format(coco_img_path))
                except:
                    raise e

            
            ## Uncomment to visualize every sample
            # ax = self.coco_dataset.showImgAnn(np.asarray(image), this_sample_anns, bbox_only=True, BGR=False)
            # # plt.show(block=False)
            # plt.savefig('fig.jpg')
            # plt.close()
            # input('here plot')

        self.logger.info('Saving annotations to disk')
        self.coco_dataset.saveAnnsToDisk()
        self.logger.info('Conversion complete!')
    ##------------------------------------------------------------------------------
    def nuscene_cat_to_coco(self, nusc_ann_name):
        """
        Convert nuscene categories to COCO cat, cat_id and supercategory

        :param nusc_ann_name (str): Nuscenes annotation name
        """
        try:
            coco_equivalent = self.NUSC_COCO_CAT_MAP[nusc_ann_name]
        except KeyError:
            return None, None, None
        coco_cat = coco_equivalent['name']
        coco_id = coco_equivalent['id']
        coco_supercat = coco_equivalent['supercategory']
        return coco_cat, coco_id, coco_supercat
    ## -------------------------------------------------------------------------
    def get_box_dist_to_cam(self, box):
            """
            Calculates the cylindrical (xy) center distance from camera to a box.
            :param box (Box): The NuScenes annotation box in camera coordinates
            :return: distance (in meters)
            """

            # Distance can be calculated directly from box center
            # Note that the y component in the camera coordinates is downward.
            ego_dist = np.sqrt(np.sum(np.array(box.center[[0,2]]) ** 2))
            return round(ego_dist,2)
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