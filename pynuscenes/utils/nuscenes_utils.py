import numpy as np
import math
import copy
from pyquaternion import Quaternion
from shapely.geometry import LineString
from nuscenes.utils.geometry_utils import view_points, box_in_image
from nuscenes.utils import splits
from pynuscenes.utils import constants as NS_C
from nuscenes.utils.data_classes import Box, LidarPointCloud, RadarPointCloud, PointCloud
from nuscenes.eval.detection.data_classes import DetectionBox


def map_pointcloud_to_camera(pointcloud, cam_cs_record, cam_pose_record,
                            pointsensor_pose_record=None, coordinates='vehicle'):
    """
    Map the point cloud to camera coordinates.
    
    :param pc (PointCloud): point cloud in vehicle/global coordinates
    :param cam_cs_record (dict): Camera calibrated sensor record
    :param cam_pose_record (dict): Ego vehicle pose record for the timestamp of the camera
    :param pointsensor_pose_record (dict): Ego vehicle pose record for the timestamp of the point sensor
    :param coordinates (str): Point cloud coordinates ('vehicle', 'global') 
    :return points (nparray), depth: Points in camera's coordinates
    """
    pc = copy.deepcopy(pointcloud)
    
    ## Transform point cloud into the camera coordinates via global
    ## First step: transform to global frame if in vehicle frame
    if coordinates == 'vehicle':
        assert pointsensor_pose_record is not None, 'Erroe: pointsensor_pose_record is required.'
        pc = vehicle_to_global(pc, pointsensor_pose_record)
    
    ## Second step: transform to ego vehicle frame for the timestamp of the image
    pc = global_to_vehicle(pc, cam_pose_record)
    
    ## Third step: transform into the camera.
    pc = vehicle_to_sensor(pc, cam_cs_record)
    
    ## Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    return pc, depths
##------------------------------------------------------------------------------
def map_pointcloud_to_image(pointcloud, cam_intrinsic, img_shape=(1600,900)):
    """
    Map point cloud from camera coordinates to the image
    
    :param pc (PointCloud): point cloud in camera coordinates
    :param cam_cs_record (dict): Camera calibrated sensor record
    :param img_shape: shape of the image (width, height)
    :param coordinates (str): Point cloud coordinates ('vehicle', 'global') 
    :return points (nparray), depth, mask: Mapped and filtered points with depth and mask
    """
    pc = copy.deepcopy(pointcloud)

    if isinstance(pc, LidarPointCloud) or isinstance(pc, RadarPointCloud):
        points = pc.points[:3,:]
    else:
        points = pc

    (width, height) = img_shape
    depths = points[2, :]
    
    ## Take the actual picture
    points = view_points(points[:3, :], cam_intrinsic, normalize=True)

    ## Remove points that are either outside or behind the camera. 
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < width - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < height - 1)
    points = points[:, mask]
    depths = depths[mask]
    return points, depths, mask
##------------------------------------------------------------------------------
def map_annotation_to_camera(annotation, cam_cs_record, cam_pose_record,
                            ref_pose_record=None, coordinates='vehicle'):
    """
    Given an annotation Box and camera information, map the box to the image.
    
    :param annotation (Box): annotation box in vehicle or global coordinates
    :param cam_cs_record (dict): Camera calibrated sensor record
    :param cam_pose_record (dict): Ego vehicle pose record for the timestamp of the camera
    :param ref_pose_record (dict): reference pose record used for transforming 
        anns from global to vehicle coordinate system
    :param coordinates (str): current point cloud coordinates ('vehicle', 'global') 
    :return ann (Box): Mapped annotation box
    """
    ann = copy.deepcopy(annotation)
    ## Transform ann into the camera coordinates via global
    ## First step: transform to global frame if in vehicle frame
    if coordinates == 'vehicle':
        assert ref_pose_record is not None, 'Erroe: ref_pose_record is required.'
        ann = vehicle_to_global(ann, ref_pose_record)
    
    ## Second step: transform to ego vehicle frame for the timestamp of the image
    ann = global_to_vehicle(ann, cam_pose_record)
    
    ## Third step: transform into the camera.
    ann = vehicle_to_sensor(ann, cam_cs_record)

    return ann
##------------------------------------------------------------------------------
def box_3d_to_2d_simple(box, view, imsize, mode='xywh'):
        """
        Projects 3D box into image FOV.
        
        :param box: 3D box in camera reference frame.
        :param view: <np.float: 3, 4>. Projection matrix.
        :param mode (str): 2D bbox fotmat: 'xywh' or 'xyxy'
        :param imsize: (width, height). Image size.
        :return bbox: Bounding box in image plane or None if box is not in the image.
        """

        # Create a new box.
        box = box.copy()

        # Check that some corners are inside the image.
        corners = np.array([corner for corner in box.corners().T if corner[2] > 0]).T
        if len(corners) == 0:
            return None

        # Project corners that are in front of the camera to 2d to get bbox in pixel coords.
        imcorners = view_points(corners, view, normalize=True)[:2]
        bbox = (np.min(imcorners[0]), np.min(imcorners[1]), np.max(imcorners[0]), np.max(imcorners[1]))

        # Crop bbox to prevent it extending outside image.
        bbox_crop = tuple(max(0, b) for b in bbox)
        bbox_crop = (min(imsize[0], bbox_crop[0]),
                     min(imsize[0], bbox_crop[1]),
                     min(imsize[0], bbox_crop[2]),
                     min(imsize[1], bbox_crop[3]))

        # Detect if a cropped box is empty.
        # if bbox_crop[0] >= bbox_crop[2] or bbox_crop[1] >= bbox_crop[3]:
        if bbox_crop[0] >= bbox_crop[2]-1 or bbox_crop[1] >= bbox_crop[3]-1:
            return None
        
        if mode == 'xywh':
            bbox_crop = [bbox_crop[0], bbox_crop[1], 
                         abs(bbox_crop[2]-bbox_crop[0]), 
                         abs(bbox_crop[3]-bbox_crop[1])]
        return np.around(bbox_crop, 2)
##------------------------------------------------------------------------------
# def box_3d_to_2d(box, view, imsize, mode='xywh'):
#     """ # TODO: currently not working properly
#     Convert a 3d annotation box to its 2D bounding box equivalent

#     :param box (Box): Annotation box in camera coordinate system
#     :param view (nparray): Camera intrinsics matrix
#     :param imsize (width, height): Image size in pixels
#     :param wlh_factor (float): Multiply w, l, h by a factor to inflate or deflate the box.
#     :return bbox (nparray): 2D bounding box (xywh or xyxy)
#     """

#     corners_3d = box.corners()
#     corners_img = view_points(corners_3d, view, normalize=True)[:2,:]

#     # Find corners that are outside image boundaries
#     invisible = np.logical_or(corners_img[0, :] < 0, corners_img[0, :] > imsize[0])
#     invisible = np.logical_or(invisible, corners_img[1, :] > imsize[1])
#     invisible = np.logical_or(invisible, corners_img[1, :] < 0)
#     ind_invisible = [i for i, x in enumerate(invisible) if x]
#     corner_2d_visible = np.delete(corners_img, ind_invisible, 1)
    
#     neighbor_map = {0: [1,3,4], 1: [0,2,5], 2: [1,3,6], 3: [0,2,7],
#                     4: [0,5,7], 5: [1,4,6], 6: [2,5,7], 7: [3,4,6]}
#     border_lines = [[(0,imsize[1]),(0,0)],
#                     [(imsize[0],0),(imsize[0],imsize[1])],
#                     [(imsize[0],imsize[1]),(0,imsize[1])],
#                     [(0,0),(imsize[0],0)]]

#     # Find intersections with boundary lines
#     for ind in ind_invisible:
#         # intersections = []
#         invis_point = (corners_img[0, ind], corners_img[1, ind])
#         for i in neighbor_map[ind]:
#             if i in ind_invisible:
#                 # Both corners outside image boundaries, ignore them
#                 continue

#             nbr_point = (corners_img[0,i], corners_img[1,i])
#             line = LineString([invis_point, nbr_point])
#             for borderline in border_lines:
#                 intsc = line.intersection(LineString(borderline))
#                 if not intsc.is_empty:
#                     corner_2d_visible = np.append(corner_2d_visible, np.asarray([[intsc.x],[intsc.y]]), 1)
#                     break

#     # Construct a 2D box covering the whole object
#     x_min, y_min = np.amin(corner_2d_visible, 1)
#     x_max, y_max = np.amax(corner_2d_visible, 1)

#     # Get the box corners
#     # corners_img = np.array([[x_max, x_max, x_min, x_min],
#     #                     [y_max, y_min, y_min, y_max]])

#     # Convert to the MS COCO bbox format
#     # bbox = [corners_img[0,3], corners_img[1,3],
#     #         corners_img[0,0]-corners_img[0,3],corners_img[1,1]-corners_img[1,0]]
#     if mode == 'xyxy':
#         bbox = [x_min, y_min, x_max, y_max]
#     elif mode == 'xywh':
#         bbox = [x_min, y_min, abs(x_max-x_min), abs(y_max-y_min)]
#     else: 
#         raise Exception("mode of '{}' is not supported".format(mode))

#     return bbox
##------------------------------------------------------------------------------
def boxes_in_image(boxes, cam_cs_record, img_shape=(1600,900)):
    """
    Check if list of annotations are inside an image

    :param boxes: annotation boxes in camera coordinates
    :param cam_cs_record: calibrated sensor record of the camera
    :param img_shape: shape of the image (width, height)
    """
    assert isinstance(boxes[0], Box)
    camera_intrinsic = np.array(cam_cs_record['camera_intrinsic'])
    visible_boxes = []
    mask = []
    for box in boxes:
        if box_in_image(box, camera_intrinsic, img_shape):
            visible_boxes.append(box)
            mask.append(True)
        else:
            mask.append(False)
    return visible_boxes, mask
##------------------------------------------------------------------------------
def split_scenes(scenes, split):
    """
    Get the list of scenes in a split
    
    :param scenes (list): list of all scenes from nuscene
    :param split (str): split name
    :return scene_list(list): list of scene tokens in the split
    """
    scene_split_names = splits.create_splits_scenes()[split]
    scenes_list = []        
    for scene in scenes:
        #NOTE: mini train and mini val are subsets of train and val
        if scene['name'] in scene_split_names:
            scenes_list.append(scene['token'])
    return scenes_list
##------------------------------------------------------------------------------
def global_to_vehicle(data, pose_record):
    """
    Transform points/boxes from global to vehicle coordinates

    :param obj: An object of PointCloud or Box class
    :param pose_record (dict): Vehicle ego-pose dictionary
    :return obj: Object in the vehicle coordinate system
    """
    obj = copy.deepcopy(data)    
    translation_matrix = -np.array(pose_record['translation'])

    if isinstance(obj, PointCloud):
        rotation_matrix = Quaternion(pose_record['rotation']).rotation_matrix.T
    elif isinstance(obj, Box):
        rotation_matrix = Quaternion(pose_record['rotation']).inverse
    elif isinstance(obj, DetectionBox):
        quat = Quaternion(pose_record['rotation']).inverse

        obj.translation += translation_matrix

        obj.translation = np.dot(quat.rotation_matrix, np.array(obj.translation))
        obj.rotation = quat * Quaternion(obj.rotation)
        obj.velocity = np.dot(quat.rotation_matrix, np.array([obj.velocity[0], obj.velocity[1], 0]))

        obj.translation = obj.translation.tolist()
        obj.rotation = obj.rotation.elements.tolist()
        velTmp = obj.velocity.tolist()
        obj.velocity = [velTmp[0], velTmp[1]]
        return obj
    else:
        raise TypeError('Input must be a PointCloud or Box object. got {}'.format(type(obj)))
    
    ## TODO: check for correctness then uncomment
#     elif isinstance(pc, np.ndarray):
#         ## Transform from global to vehicle
#         for i in range(3):
#             pc[i, :] = pc[i, :] + np.array(-np.array(ego_pose['translation']))[i]
#         pc[:3, :] = np.dot(Quaternion(ego_pose['rotation']).rotation_matrix.T, pc[:3, :])

    ## Apply the transforms
    obj.translate(translation_matrix)
    if isinstance(obj, RadarPointCloud):
        velocity = np.array(data.points[8:10,:])
        velocity = np.vstack((velocity, np.zeros((1,data.points.shape[1]))))
        data.points[8:10,:] = np.dot(rotation_matrix, velocity)[:2,:]
    obj.rotate(rotation_matrix)
    
    return obj
##------------------------------------------------------------------------------
def vehicle_to_global(data, pose_record):
    """
    Transform points/boxes from vehicle to global coordinates

    :param obj: An object of PointCloud or Box class
    :param pose_record (dict): Vehicle ego-pose dictionary
    :return obj: Object in the global coordinate system
    """
    obj = copy.deepcopy(data)
    translation_matrix = np.array(pose_record['translation'])
    
    if isinstance(obj, PointCloud):
        rotation_matrix = Quaternion(pose_record['rotation']).rotation_matrix
    elif isinstance(obj, Box):
        rotation_matrix = Quaternion(pose_record['rotation'])
    elif isinstance(obj, DetectionBox):
        quat = Quaternion(pose_record['rotation'])

        obj.translation = np.dot(quat.rotation_matrix, np.array(obj.translation))
        obj.rotation = quat * Quaternion(obj.rotation)
        obj.velocity = np.dot(quat.rotation_matrix, np.array([obj.velocity[0], obj.velocity[1], 0]))

        obj.translation += translation_matrix

        obj.translation = obj.translation.tolist()
        obj.rotation = obj.rotation.elements.tolist()
        velTmp = obj.velocity.tolist()
        obj.velocity = [velTmp[0], velTmp[1]]
        return obj
    else:
        raise TypeError('Input must be a PointCloud or Box object. got {}'.format(type(obj)))
    
    ## Apply the transforms
    obj.rotate(rotation_matrix)
    if isinstance(obj, RadarPointCloud):
        velocity = np.array(data.points[8:10,:])
        velocity = np.vstack((velocity, np.zeros((1,data.points.shape[1]))))
        data.points[8:10,:] = np.dot(rotation_matrix, velocity)[:2,:]
    obj.translate(translation_matrix)
    
    return obj
##------------------------------------------------------------------------------
def vehicle_to_sensor(data, cs_record):
    """
    Transform points/boxes from vehicle to sensor coordinates

    :param obj: An object of PointCloud or Box class
    :param cs_record (dict): Calibrated sensor record dictionary
    :return obj: Object in the sensor coordinate system
    """
    obj = copy.deepcopy(data)
    translation_matrix = -np.array(cs_record['translation'])

    if isinstance(obj, PointCloud):
        rotation_matrix = Quaternion(cs_record['rotation']).rotation_matrix.T
    elif isinstance(obj, Box):
        rotation_matrix = Quaternion(cs_record['rotation']).inverse
    else:
        raise TypeError('Input must be a PointCloud or Box object')
        
    ## TODO: check for correctness then uncomment
#     elif isinstance(pc, np.ndarray):
#         for i in range(3):
#             pc[i, :] = pc[i, :] - np.array(cs_record['translation'])[i]
#         pc[:3, :] = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, pc[:3, :])

    ## Apply the transforms
    obj.translate(translation_matrix)
    if isinstance(obj, RadarPointCloud):
        velocity = np.array(data.points[8:10,:])
        velocity = np.vstack((velocity, np.zeros((1,data.points.shape[1]))))
        data.points[8:10,:] = np.dot(rotation_matrix, velocity)[:2,:]
    obj.rotate(rotation_matrix)
    
    return obj
##------------------------------------------------------------------------------
def sensor_to_vehicle(data, cs_record):
    """
    Transform points/boxes from sensor to vehicle coordinates

    :param obj: An object of PointCloud or Box class
    :param cs_record (dict): Calibrated sensor record dictionary
    :return obj: Object in the vehicle coordinate system
    """
    obj = copy.deepcopy(data)
    translation_matrix = np.array(cs_record['translation'])

    if isinstance(obj, PointCloud):
        rotation_matrix = Quaternion(cs_record['rotation']).rotation_matrix
    elif isinstance(obj, Box):
        rotation_matrix = Quaternion(cs_record['rotation'])
    else:
        raise TypeError('Input must be a PointCloud or Box object')

    ## Apply the transforms
    obj.rotate(rotation_matrix)
    if isinstance(obj, RadarPointCloud):
        velocity = np.array(data.points[8:10,:])
        velocity = np.vstack((velocity, np.zeros((1,data.points.shape[1]))))
        data.points[8:10,:] = np.dot(rotation_matrix, velocity)[:2,:]
    obj.translate(translation_matrix)

    return obj
##------------------------------------------------------------------------------
def get_box_dist(box, pose_record):
        """
        Calculates the cylindrical (xy) center distance from ego vehicle to each box.
        
        :param box (Box): The NuScenes annotation box in global coordinates
        :param pose_record: Ego pose record from the LIDAR
        :return: distance (in meters)
        """
        # Both boxes and ego pose are given in global coord system, so distance 
        # can be calculated directly. Note that the z component of the ego pose is 0.
        ego_translation = (box.center[0] - pose_record['translation'][0],
                           box.center[1] - pose_record['translation'][1],
                           box.center[2] - pose_record['translation'][2])
        ego_dist = np.sqrt(np.sum(np.array(ego_translation[:2]) ** 2))
        return round(ego_dist,2)
##------------------------------------------------------------------------------
def bbox_to_corners(bboxes):
    """ # TODO: check compatibility
    Convert 3D bounding boxes in [x,y,z,w,l,h,ry] format to [x,y,z] coordinates
    of the corners
    
    :param bboxes: input boxes np.ndarray <N,7>
    :return corners: np.ndarray <N,3,8> where x,y,z is along each column
    """
    x = np.expand_dims(bboxes[:,0], 1)
    y = np.expand_dims(bboxes[:,1], 1)
    z = np.expand_dims(bboxes[:,2], 1)

    w = np.expand_dims(bboxes[:,3], 1)
    l = np.expand_dims(bboxes[:,4], 1)
    h = np.expand_dims(bboxes[:,5], 1)
    
    yaw_angle = bboxes[:,6]

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = (l / 2) * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    y_corners = (w / 2) * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = (h / 2) * np.array([1,  1, -1, -1,  1,  1, -1, -1])
    corners = np.dstack((x_corners, y_corners, z_corners))
    # Rotate
    for i, box in enumerate(corners):
        rotation_quat = Quaternion(axis=(0, 0, 1), angle=yaw_angle[i])
        corners[i,:,:] = np.dot(rotation_quat.rotation_matrix, box.T).T

    # Translate
    corners[:,:,0] += x
    corners[:,:,1] += y
    corners[:,:,2] += z

    corners = np.swapaxes(corners, 1,2)
    return corners
##------------------------------------------------------------------------------
def corners3d_to_image(corners, cam_cs_record, img_shape):
    """ # TODO: check compatibility
    Return the 2D box corners mapped to the image plane
    
    :param corners (np.array <N, 3, 8>)
    :param cam_cs_record (dict): calibrated sensor record of a camera dictionary from nuscenes dataset
    :param img_shape (tuple<width, height>)
    :return (ndarray<N,2,8>, list<N>)
    """
    cornerList = []
    mask = []
    for box_corners in corners:
        box_corners = pc_to_sensor(box_corners, cam_cs_record)
        this_box_corners = view_points(box_corners, np.array(cam_cs_record['camera_intrinsic']), normalize=True)[:2, :]
        
        visible = np.logical_and(this_box_corners[0, :] > 0, this_box_corners[0, :] < img_shape[0])
        visible = np.logical_and(visible, this_box_corners[1, :] < img_shape[1])
        visible = np.logical_and(visible, this_box_corners[1, :] > 0)
        visible = np.logical_and(visible, box_corners[2, :] > 1)
        in_front = box_corners[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.
        isVisible = any(visible) and all(in_front)
        mask.append(isVisible)
        if isVisible:
            cornerList.append(this_box_corners)
    return np.array(cornerList), mask
##------------------------------------------------------------------------------
def quaternion_to_ry(quat: Quaternion):
    v = np.dot(quat.rotation_matrix, np.array([1,0,0]))
    yaw = np.arctan2(v[1], v[0])
    return yaw
