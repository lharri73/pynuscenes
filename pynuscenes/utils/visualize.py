################################################################################
## Date Created  : Thu Jun 13 2019                                            ##
## Authors       : Landon Harris, Ramin Nabati                                ##
## Last Modified : Sat Jun 15 2019                                            ##
## Copyright (c) 2019                                                         ##
################################################################################

import cv2
from pynuscenes.utils import constants as _C
import numpy as np
from pynuscenes.nuscenes_dataset import NuscenesDataset
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import copy
from mayavi.mlab import *

def show_sample_data(sample):
    """
    Render the data from all sensors in a single sample
    :param sample: sample dictionary returned from nuscenes_db
    """
    ## At this point, the point clouds are in vehicle coordinates
    # map_pointcloud_to_image(sample['lidar']['points'], sample['camera'][0]['image'], sample['camera'][0]['cs_record'])
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

    image = _arrange_images(image_list)
    print(sample['lidar']['points'].points.shape)
    cv2.imshow('images', image)
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
    # pc.translate(-np.array(cam_cs_record['translation']))
    # pc.rotate(Quaternion(cam_cs_record['rotation']).rotation_matrix.T)
    pc = NuscenesDataset.pc_to_sensor(pc, cam_cs_record)
    points3d(pc.points[0, :], pc.points[1, :], pc.points[2, :], scale_factor=.25)
    input('this should be the points in camera coordinates')

    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    # Retrieve the color from the depth.
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cam_cs_record['camera_intrinsic']), normalize=True)
    # points3d(0, 0, 0, color=(1, 0, 0))
    np.set_printoptions(threshold=np.inf)
    print(points.T)
    
    points3d(points[0,:], points[1,:], points[2,:], scale_factor=25)
    input('this should be the points in camera coordinates')

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.shape[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.shape[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]
    # np.set_printoptions(threshold=np.inf)
    im = plot_points_on_image(im, points.T, coloring)
    # cv2.imshow('im1', im)
    # cv2.waitKey(0)
    return im

def plot_points_on_image(image, points, coloring):
    newPoint = [0,0]
    for i, point in enumerate(points):
        newPoint[0], newPoint[1] = int(point[0]), int(point[1])
        cv2.circle(image, tuple(newPoint), 2, (int(coloring[i]),int(coloring[i]),int(coloring[i])), -1)
    return image
