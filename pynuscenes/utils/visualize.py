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

def show_sample_data(sample):
    """
    Render the data from all sensors in a single sample
    :param sample: sample dictionary returned from nuscenes_db
    """

    top_row = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
    btm_row = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    image_list = [[], []] 

    for cam in top_row:
        image_list[0].append(sample['camera'][_C.CAMERAS[cam]]['image'])
    for cam in btm_row:
        image_list[1].append(sample['camera'][_C.CAMERAS[cam]]['image'])

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
