################################################################################
## Date Created  : Fri Jun 14 2019                                            ##
## Authors       : Landon Harris, Ramin Nabati                                ##
## Last Modified : Fri Jun 14 2019                                            ##
## Copyright (c) 2019                                                         ##
################################################################################

import numpy as np
import math
from pyquaternion import Quaternion
from nuscenes_dataset.pynuscenes.nuscenes_dataset import NuscenesDataset
from nuscenes.utils.geometry_utils import view_points
from shapely.geometry import LineString

def boxes3d_to_corners3d(boxes3d, swap=False, cam_coord=False, rotate=True):
    """
    Returns the bounding box corners.
    :param wlh_factor: Multiply w, l, h by a factor to scale the box.
    :return: <n, 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    corners_list = []
    for box in boxes3d:
        corners_list.append(bbox_to_corners(box, camera_coord=cam_coord))
    res = np.array(corners_list)
    if swap:
        try:
            res = np.swapaxes(res, 1,2)
        except ValueError:
            return res
    return res

def bbox_to_corners(bbox, camera_coord=False):
    """
    Convert a 3D bounding box in [x,y,z,w,l,h,ry] format to corners
    :return corners: (3,N) where x,y,z is along each column
    """
    x, y, z = bbox[0:3]
    w, l, h = bbox[3:6]
    yaw_angle = bbox[6]

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    if camera_coord:
        rotation_quat = Quaternion(axis=(0, 1, 0), angle=yaw_angle)
    else:
        rotation_quat = Quaternion(axis=(0, 0, 1), angle=yaw_angle)
    corners = np.dot(rotation_quat.rotation_matrix, corners)

    # Translate
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners

def quaternion_to_ry(quat: Quaternion):
    v = np.dot(quat.rotation_matrix, np.array([1,0,0]))
    yaw = np.arctan2(v[1], v[0])
    return yaw

def corners3d_to_image(corners, cam_cs_record, img_shape):
    """
    :param corners: np.array <n, 8, 3>
    :param cam_cs_record: calibrated sensor record of a camera dictionary from nuscenes dataset
    :param img_shape [width, height]
    """
    cornerList = []
    for box_corners in corners:

        NuscenesDataset.pc_to_sensor(box_corners, cam_cs_record)
        this_box_corners = view_points(box_corners, np.array(cam_cs_record['camera_intrinsic']), normalize=True)[:2, :]

        if np.any(np.isinf(this_box_corners)) or np.any(np.isnan(this_box_corners)):
            continue

        visible = np.logical_and(this_box_corners[0, :] > 0, this_box_corners[0, :] < img_shape[0])
        visible = np.logical_and(visible, this_box_corners[1, :] < img_shape[1])
        visible = np.logical_and(visible, this_box_corners[1, :] > 0)
        visible = np.logical_and(visible, box_corners[2, :] > 1)

        in_front = box_corners[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

        isVisible = any(visible) and all(in_front)
        if isVisible:
            cornerList.append(this_box_corners)
    return np.array(cornerList)

def box_corners_to_2dBox(corners_2d, imsize, mode='xywh'):
        """
        Convert the 3d box to the 2D bounding box in COCO format (x,y,h,w)

        :param view
        :param imsize: Image size in pixels
        :param wlh_factor: <float>. Multiply w, l, h by a factor to inflate or deflate the box.
        :return: <np.float: 2, 4>. Corners of the 2D box
        """
        bboxes = []
        for corner_2d in corners_2d:
            neighbor_map = {0: [1,3,4], 1: [0,2,5], 2: [1,3,6], 3: [0,2,7],
                            4: [0,5,7], 5: [1,4,6], 6: [2,5,7], 7: [3,4,6]}
            border_lines = [[(0,imsize[1]),(0,0)],
                            [(imsize[0],0),(imsize[0],imsize[1])],
                            [(imsize[0],imsize[1]),(0,imsize[1])],
                            [(0,0),(imsize[0],0)]]

            # Find corners that are outside image boundaries
            invisible = np.logical_or(corner_2d[0, :] < 0, corner_2d[0, :] > imsize[0])
            invisible = np.logical_or(invisible, corner_2d[1, :] > imsize[1])
            invisible = np.logical_or(invisible, corner_2d[1, :] < 0)
            ind_invisible = [i for i, x in enumerate(invisible) if x]
            corner_2d_visible = np.delete(corner_2d, ind_invisible, 1)

            # Find intersections with boundary lines
            for ind in ind_invisible:
                # intersections = []
                invis_point = (corner_2d[0, ind], corner_2d[1, ind])
                for i in neighbor_map[ind]:
                    if i in ind_invisible:
                        # Both corners outside image boundaries, ignore them
                        continue

                    nbr_point = (corner_2d[0,i], corner_2d[1,i])
                    line = LineString([invis_point, nbr_point])
                    for borderline in border_lines:
                        intsc = line.intersection(LineString(borderline))
                        if not intsc.is_empty:
                            corner_2d_visible = np.append(corner_2d_visible, np.asarray([[intsc.x],[intsc.y]]), 1)
                            break

            # Construct a 2D box covering the whole object
            x_min, y_min = np.amin(corner_2d_visible, 1)
            x_max, y_max = np.amax(corner_2d_visible, 1)

            # Get the box corners
            corner_2d = np.array([[x_max, x_max, x_min, x_min],
                                [y_max, y_min, y_min, y_max]])

            # Convert to the MS COCO bbox format
            # bbox = [corner_2d[0,3], corner_2d[1,3],
            #         corner_2d[0,0]-corner_2d[0,3],corner_2d[1,1]-corner_2d[1,0]]
            if mode == 'xyxy':
                bbox = [x_min, y_min, x_max, y_max]
            elif mode == 'xywh':
                bbox = [x_min, y_min, abs(x_max-x_min), abs(y_max-y_min)]
            else: 
                raise Exception("mode of '{}'' is not supported".format(mode))
            bboxes.append(bbox)

        return bboxes