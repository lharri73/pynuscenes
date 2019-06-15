################################################################################
## Date Created  : Thu Jun 13 2019                                            ##
## Authors       : Landon Harris, Ramin Nabati                                ##
## Last Modified : Fri Jun 14 2019                                            ##
## Copyright (c) 2019                                                         ##
################################################################################

import cv2
from pynuscenes.utils import constants as _C
import numpy as np
from pynuscenes.nuscenes_dataset import NuscenesDataset
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

def show_sample_data(sample):
    """
    Render the data from all sensors in a single sample
    :param sample: sample dictionary returned from nuscenes_db
    """
    map_pointcloud_to_image(sample['lidar']['points'], sample['camera'][0]['image'], sample['camera'][0]['cs_record'])
    top_row = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
    btm_row = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    image_list = [[], []] 

    for cam in top_row:
        image = map_pointcloud_to_image(sample['lidar']['points'],
         sample['camera'][_C.CAMERAS[cam]]['image'], 
         sample['camera'][_C.CAMERAS[cam]]['cs_record'])
        image_list[0].append(image)
    for cam in btm_row:
        image = map_pointcloud_to_image(sample['lidar']['points'],
         sample['camera'][_C.CAMERAS[cam]]['image'], 
         sample['camera'][_C.CAMERAS[cam]]['cs_record'])
        image_list[1].append(image)

    print(len(image_list[0]))
    image = _arrange_images(image_list)

        
    cv2.imshow('image', image)
    cv2.waitKey(0)

def _arrange_images(image_list: list, im_size: tuple=(640,360)) -> np.ndarray:
    """
    Arranges multiple images into a single image
    :param image_list: list rows where a row is a list of images
    :param image_size: new size of the images
    :return: the new image as a numpy array
    """
    rows = []
    for row in image_list:
        rows.append(np.hstack((cv2.cvtColor(cv2.resize(pic, im_size), cv2.COLOR_RGB2BGR))\
             for pic in row))
    image = np.vstack((row) for row in rows)
    return image

def map_pointcloud_to_image(pc, im, cam_cs_record):
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """
    ## Transform into the camera.
    pc.translate(-np.array(cam_cs_record['translation']))
    pc.rotate(Quaternion(cam_cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    # Retrieve the color from the depth.
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cam_cs_record['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.shape[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.shape[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]
    np.set_printoptions(threshold=np.inf)
    im = plot_points_on_image(im, points.T)

    return im

def plot_points_on_image(image, points):
    newPoint = [0,0]
    for point in points:
        newPoint[0], newPoint[1] = int(point[0]), int(point[1])
        cv2.circle(image, tuple(newPoint), 2, (255,0,0), -1)
    return image