# nuScenes dev-kit.
# Code written by Sergi Adipraja Widjaja, 2019.

"""
Export 2D annotations (xmin, ymin, xmax, ymax) from re-projections of our annotated 3D bounding boxes to a .json file.

Note: Projecting tight 3d boxes to 2d generally leads to non-tight boxes.
      Furthermore it is non-trivial to determine whether a box falls into the image, rather than behind or around it.
      Finally some of the objects may be occluded by other objects, in particular when the lidar can see them, but the
      cameras cannot.
"""

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points

import numpy as np
import json
import argparse
import os

from typing import List, Tuple, Union
from pyquaternion.quaternion import Quaternion
from collections import OrderedDict, defaultdict
from tqdm import tqdm
from shapely.geometry import MultiPoint, box


def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def get_2d_boxes(sample_data_token: str, visibilities: List[str], mode: str) -> List[OrderedDict]:
    """
    Get the 2D annotation records for a given `sample_data_token`.
    :param sample_data_token: Sample data token belonging to a camera keyframe.
    :param visibilities: Visibility filter.
    :param mode: 'xywh' or 'xyxy'
    :return: List of 2D annotation record that belongs to the input `sample_data_token`
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError('The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
    ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec['visibility_token'] in visibilities)]

    sd_annotations = []

    for ann_rec in ann_recs:
        # Get the 3D box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Calculate distance from vehicle to box
        ego_translation = (box.center[0] - pose_rec['translation'][0],
                           box.center[1] - pose_rec['translation'][1],
                           box.center[2] - pose_rec['translation'][2])
        ego_dist = np.sqrt(np.sum(np.array(ego_translation[:2]) ** 2))
        dist = round(ego_dist,2)

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
        if final_coords is None:
            continue

        min_x, min_y, max_x, max_y = final_coords
        if mode == 'xywh':
            bbox = [min_x, min_y, max_x-min_x, max_y-min_y]
        else:
            bbox = [min_x, min_y, max_x, max_y]
        
        ## Generate 2D record to be included in the .json file.
        ann_2d = {}
        ann_2d['sample_data_token'] = sample_data_token
        ann_2d['bbox'] = np.around(bbox, 2).tolist()
        ann_2d['distance'] = dist
        ann_2d['category_name'] = ann_rec['category_name']
        ann_2d['num_lidar_pts'] = ann_rec['num_lidar_pts']
        ann_2d['num_radar_pts'] = ann_rec['num_radar_pts']
        ann_2d['visibility_token'] = ann_rec['visibility_token']

        sd_annotations.append(ann_2d)

    return sd_annotations


def main(args):
    """Generates 2D re-projections of the 3D bounding boxes present in the dataset."""

    print("Generating 2D reprojections of the nuScenes dataset")

    # Get tokens for all camera images.
    sample_data_camera_tokens = [s['token'] for s in nusc.sample_data if (s['sensor_modality'] == 'camera') and
                                 s['is_key_frame']]

    # Loop through the records and apply the re-projection algorithm.
    reprojections = {}
    for token in tqdm(sample_data_camera_tokens):
        reprojections[token] = get_2d_boxes(token, args.visibilities, args.box_mode)

        # ## TESTING----------
        # import matplotlib.pyplot as plt
        # from PIL import Image
        # import io
        # from pynuscenes.utils.visualize import draw_xywh_bbox

        # cam_rec = nusc.get('sample_data', token)
        # filename = os.path.join(args.dataroot, cam_rec['filename'])
        # figure, ax = plt.subplots(1, 1, figsize=(16, 9))

        # with open(filename, 'rb') as f:
        #     image_str = f.read()
        # image = np.array(Image.open(io.BytesIO(image_str)))
        # anns = reprojections[token]
        # ax.imshow(image)
        # for ann in anns:
        #     bbox = ann['bbox']
        #     ax = draw_xywh_bbox(bbox, ax)
            
        # plt.savefig('this')
        # plt.close()
        # input('here')
        # ##------------------

    # Save to a .json file.
    dest_path = os.path.join(args.dataroot, args.version)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    # with open(os.path.join(args.dataroot, args.version, args.filename), 'w') as fh:
    with open(args.filename, 'w') as fh:
        # json.dump(reprojections, fh, sort_keys=True, indent=4)
        json.dump(reprojections, fh, indent=4)

    print("Saved the 2D re-projections under {}".format(os.path.join(args.dataroot, args.version, args.filename)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export 2D annotations from reprojections to a .json file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='../../data/nuscenes', help="Path where nuScenes is saved.")
    parser.add_argument('--version', type=str, default='v1.0-mini', help='Dataset version.')
    parser.add_argument('--filename', type=str, default='image_annotations.json', help='Output filename.')
    parser.add_argument('--visibilities', type=str, default=['', '1', '2', '3', '4'],
                        help='Visibility bins, the higher the number the higher the visibility.', nargs='+')
    parser.add_argument('--box_mode', type=str, default='xywh', choices=['xywh', 'xyxy'])
    args = parser.parse_args()

    nusc = NuScenes(dataroot=args.dataroot, version=args.version)
    main(args)