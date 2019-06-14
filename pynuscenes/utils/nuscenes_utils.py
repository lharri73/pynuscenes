################################################################################
## Date Created  : Fri Jun 14 2019                                            ##
## Authors       : Landon Harris, Ramin Nabati                                ##
## Last Modified : Fri Jun 14 2019                                            ##
## Copyright (c) 2019                                                         ##
################################################################################

import numpy as np
import math
from scipy.spatial import Delaunay
import scipy
# from config import cfg
from pyquaternion import Quaternion
# import torch

def cls_type_to_id(cls_type):
    type_to_id = {
        'animal':                               0,
        'human.pedestrian.adult':               1,
        'human.pedestrian.child':               2,
        'human.pedestrian.construction_worker': 3,
        'human.pedestrian.personal_mobility':   4,
        'human.pedestrian.police_officer':      5,
        'human.pedestrian.stroller':            6,
        'human.pedestrian.wheelchair':          7,
        'movable_object.barrier':               8,
        'movable_object.debris':                9,
        'movable_object.pushable_pullable':     10,
        'movable_object.trafficcone':           11,
        'vehicle.bicycle':                      12,
        'vehicle.bus.bendy':                    13,
        'vehicle.bus.rigid':                    14,
        'vehicle.car':                          15,
        'vehicle.construction':                 16,
        'vehicle.emergency.ambulance':          17,
        'vehicle.emergency.police':             18,
        'vehicle.motorcycle':                   19,
        'vehicle.trailer':                      20,
        'vehicle.truck':                        21,
        'static_object.bicycle_rack':           22
    }

    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, properties):
        label = properties
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.trucation = float(label[1])                # ignored
        self.occlusion = float(label[2])                # 0:0-40% 1:40-60% 2:60-80% 3:80-100% visible #NOTE: This takes ALL cameras into account
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.pos = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.pos)
        self.ry = label[14]
        self.level_str = None
        self.num_points = properties[15]
        self.level = self.get_obj_level()
        self.box = label[16]

    def get_obj_level(self):
        if self.num_points <= cfg.OBJECT_DIFFICULTY[0]:
            self.level_str = 'Easy'
            return 1  # Easy
        elif self.num_points >= cfg.OBJECT_DIFFICULTY[1]:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'Moderate'
            return 3  # Hard

    # def generate_corners3d(self):
    #     """
    #     generate corners3d representation for this object
    #     :return corners_3d: (8, 3) corners of box3d in camera coord
    #     """
    #     raise NotImplementedError("this will not generate the correct corners")
    #     l, h, w = self.l, self.h, self.w
    #     x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    #     y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    #     z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    #     R = np.array([[np.cos(self.ry), -np.sin(self.ry), 0],
    #                   [np.sin(self.ry), np.cos(self.ry), 0],
    #                   [0, 0, 1]])
    #     corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
    #     corners3d = np.dot(R, corners3d).T
    #     corners3d = corners3d + self.pos
    #     return corners3d

    # def to_bev_box2d(self, oblique=True, voxel_size=0.1):
    #     """
    #     :param bev_shape: (2) for bev shape (h, w), => (y_max, x_max) in image
    #     :param voxel_size: float, 0.1m
    #     :param oblique:
    #     :return: box2d (4, 2)/ (4) in image coordinate
    #     """
    #     if oblique:
    #         corners3d = self.generate_corners3d()
    #         xz_corners = corners3d[0:4, [0, 2]]
    #         box2d = np.zeros((4, 2), dtype=np.int32)
    #         box2d[:, 0] = ((xz_corners[:, 0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
    #         box2d[:, 1] = Object3d.BEV_SHAPE[0] - 1 - ((xz_corners[:, 1] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
    #         box2d[:, 0] = np.clip(box2d[:, 0], 0, Object3d.BEV_SHAPE[1])
    #         box2d[:, 1] = np.clip(box2d[:, 1], 0, Object3d.BEV_SHAPE[0])
    #     else:
    #         box2d = np.zeros(4, dtype=np.int32)
    #         # discrete_center = np.floor((self.pos / voxel_size)).astype(np.int32)
    #         cu = np.floor((self.pos[0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
    #         cv = Object3d.BEV_SHAPE[0] - 1 - ((self.pos[2] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
    #         half_l, half_w = int(self.l / voxel_size / 2), int(self.w / voxel_size / 2)
    #         box2d[0], box2d[1] = cu - half_l, cv - half_w
    #         box2d[2], box2d[3] = cu + half_l, cv + half_w

    #     return box2d

    # def to_str(self):
    #     print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
    #                  % (self.cls_type, self.trucation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
    #                     self.pos, self.ry)
    #     return print_str


def annotations_to_objects(nusc, boxes):
    #TODO: verify these difficulty Levels
    difficultyLevels = {
        '1': 3,
        '2': 2,
        '3': 1,
        '4': 0
    }

    objects = []

    # boxes = box_to_kittiRect(boxes)
    for box in boxes:
        stringLevel = nusc.get('visibility', nusc.get('sample_annotation', box.token)['visibility_token'])['token']
        level = difficultyLevels[stringLevel]
        num_points = nusc.get('sample_annotation', box.token)['num_lidar_pts']
        theseProperties = [
            box.name,                                       #0: label of box
            0,                                              #1: We are ignoring truncation
            level,                                          #2: level of visibility (replacing occlusion)
            0,                                              #3: ignoring alpha
                                                            ##2D Bounding Box IN PIXEL COORDINATES
            0,                                              #4: left bbox
            0,                                              #5: top bbox
            0,                                              #6: right bbox
            0,                                              #7: bottom bbox
                                                            ##Dimensions of 3d bounding box (in meters)
            box.wlh[0],                                     #8:  height
            box.wlh[1],                                     #9:  width
            box.wlh[2],                                     #10: length
                                                            ##position (in LIDAR coordinates) NOTE: this is different from the KITTI dataset
            box.center[0],                                  #11: x position
            box.center[1],                                  #12: y position
            box.center[2],                                  #13: z position
            quaternion_to_ry(box.orientation),              #14: rotation about y axis (-pi,pi)
            num_points,                                     #15: number of lidar points in the annotation
            box                                             #16: the box itself
        ]
        thisObject = Object3d(theseProperties)
        objects.append(thisObject)
    return objects



def QuaternionToEulerAngles(Q) :
    """
    Converts Quaternion to Euler rotation about the three axis
    :param: Quaternion
    :return: np array of rotations about euler axis [x,y,z]
    """
    R = Q.rotation_matrix
    theta = -np.arctan(R[1,0] / R[0,0])
    return theta

def pointcloudFromFile(file_name) -> np.array:
    """
    Loads LIDAR data from binary numpy format (stored as (x,y,z, intensity, ring index)
    :param: file_name: Path of the pointcloud file on disk
    :return: LIDAR PC as numpy array [x,y,z,intensity]
    """
    nbr_dims = 4 #ignore ring intensity
    scan = np.fromfile(file_name, dtype=np.float32)
    # Remove the ring index element
    points = scan.reshape((-1, 5))[:, :nbr_dims]
    return points

def objs_to_boxes3d(obj_list) -> [[int]]:
    """
    Takes objects, returns list of box3d's with [posX, posY, posZ, h, w, l, ry]
    :param: list of objects
    :return list of box3ds [[posX, posY, posZ], h, w, l, ry]
    """
    boxes3d = np.zeros((obj_list.__len__(), 7), dtype=np.float32)
    for k, obj in enumerate(obj_list):
        boxes3d[k, 0]   = obj.pos[0]
        boxes3d[k, 1]   = obj.pos[1]
        boxes3d[k, 2]   = obj.pos[2]
        boxes3d[k, 3]   = obj.h
        boxes3d[k, 4]   = obj.w
        boxes3d[k, 5]   = obj.l
        boxes3d[k, 6]   = obj.ry
    return boxes3d

def boxes3d_to_corners3d(boxes3d, swap=True, cam_coord=False, rotate=True):
    """
    Returns the bounding box corners.
    :param wlh_factor: Multiply w, l, h by a factor to scale the box.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
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
    # x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    # y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    # z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
    # corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    if camera_coord:
        z_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        x_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        y_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        rotation_quat = Quaternion(axis=(0, -1, 0), angle=yaw_angle)
    else:
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        rotation_quat = Quaternion(axis=(0, 0, 1), angle=yaw_angle)
    corners = np.vstack((x_corners, y_corners, z_corners))
    corners = np.dot(rotation_quat.rotation_matrix, corners)

    # Translate
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners


def ORIGINAL_bbox_to_corners(bbox, camera_coord=False):
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

def enlarge_box3d(boxes3d, extra_width):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    """
    if isinstance(boxes3d, np.ndarray):
        large_boxes3d = boxes3d.copy()
    else:
        large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += extra_width * 2
    large_boxes3d[:, 1] += extra_width
    return large_boxes3d


def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm

def get_box_mean(frame_path, class_name="vehicle.car",
                 eval_version="cvpr_2019"):
    with open(frame_path, 'rb') as f:
        nusc_frames = pickle.load(f)["frames"]
    cls_range_map = eval_detection_configs[eval_version]["class_range"]

    gt_boxes_list = []
    gt_vels_list = []
    for frame in nusc_frames:
        gt_boxes = frame["gt_boxes"]
        gt_vels = frame["gt_velocity"]
        gt_names = frame["gt_names"]
        mask = np.array([s == class_name for s in frame["gt_names"]],
                        dtype=np.bool_)
        gt_names = gt_names[mask]
        gt_boxes = gt_boxes[mask]
        gt_vels = gt_vels[mask]
        det_range = np.array([cls_range_map[n] for n in gt_names])
        det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
        mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
        mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)

        gt_boxes_list.append(gt_boxes[mask].reshape(-1, 7))
        gt_vels_list.append(gt_vels[mask].reshape(-1, 2))
    gt_boxes_list = np.concatenate(gt_boxes_list, axis=0)
    gt_vels_list = np.concatenate(gt_vels_list, axis=0)
    nan_mask = np.isnan(gt_vels_list[:, 0])
    gt_vels_list = gt_vels_list[~nan_mask]

    # return gt_vels_list.mean(0).tolist()
    return {
        "box3d": gt_boxes_list.mean(0).tolist(),
        "detail": gt_boxes_list
        # "velocity": gt_vels_list.mean(0).tolist(),
    }

##------------------------------------------------------------------------------
def get_all_box_mean(frame_path):
    det_names = set()
    for k, v in NuscenesDataset.NameMapping.items():
        if v not in det_names:
            det_names.add(v)
    det_names = sorted(list(det_names))
    res = {}
    details = {}
    for k in det_names:
        result = get_box_mean(frame_path, k)
        details[k] = result["detail"]
        res[k] = result["box3d"]
    print(json.dumps(res, indent=2))
    return details

##------------------------------------------------------------------------------
def render_nusc_result(nusc, results, sample_token):
    annos = results[sample_token]
    sample = nusc.get("sample", sample_token)
    sd_rec = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
    cs_record = nusc.get('calibrated_sensor',
                            sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    boxes = []
    for anno in annos:
        rot = Quaternion(anno["rotation"])
        box = Box(anno["translation"], anno["size"], rot, name=anno["detection_name"])
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        boxes.append(box)
    nusc.explorer.render_sample_data(sample["data"]["LIDAR_TOP"], extern_boxes=boxes, nsweeps=10)
    nusc.explorer.render_sample_data(sample["data"]["LIDAR_TOP"], nsweeps=10)

# def boxes3d_to_bev_torch(boxes3d):
#     """
#     :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
#     :return:
#         boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
#     """
#     boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

#     cu, cv = boxes3d[:, 0], boxes3d[:, 2]
#     half_l, half_w = boxes3d[:, 5] / 2, boxes3d[:, 4] / 2
#     boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
#     boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
#     boxes_bev[:, 4] = boxes3d[:, 6]
#     return boxes_bev

# def lidar_to_kittiRect(pointcloud: np.ndarray):
#     """
#     converts lidar coordinate system to kitti cam-rect coordinate system
#     :param LIDAR PC as numpy array [x,y,z,intensity]
#     :return LIDAR PC as numpy array in new coordinate system [x,y,z,intensity]
#     """
#     rot_matrix = np.array([[1,0,0],
#                           [0,0,1],
#                           [0,-1,0,]])
#     pointcloud[:,:3] = pointcloud[:,:3].dot(rot_matrix)
#     return pointcloud

# def box_to_kittiRect(boxes):
#     for box in boxes:
#         rot_matrix = np.array([[1,0,0],
#                             [0,0,1],
#                             [0,-1,0,]])
#         box.center = box.center.dot(rot_matrix)
#     return boxes

# def boxes3d_to_bev_torch(boxes3d):
#     """
#     :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
#     :return:
#         boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
#     """
#     boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

#     cu, cv = boxes3d[:, 0], boxes3d[:, 2]
#     half_l, half_w = boxes3d[:, 5] / 2, boxes3d[:, 4] / 2
#     boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
#     boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
#     boxes_bev[:, 4] = boxes3d[:, 6]
#     return boxes_bev
def quaternion_to_ry(quat: Quaternion):
    # return ry
    v = np.dot(quat.rotation_matrix, np.array([1,0,0]))

    yaw = np.arctan2(v[1], v[0])
    # print('Yaw:', yaw)
    return yaw