import pickle

import cv2
import numpy as np
from mayavi.mlab import *
from tqdm import tqdm

# from datasets.nuscenes_dataset import NuscenesDataset
import datasets.nuscenes_dataset as nsd
from utils.nuscenes_utils import *
from nuscenes.utils.data_classes import Box


def visualize_frame_data(boxes=None, radar_pc=None, lidar_pc=None, 
                         plot_origin=False, image=None, gt_boxes=None,
                         cam_coord=True):
    """
    :param anchors: list of boxes3d where a box3d is [x,y,z,w,l,h,ry]
    :param boxes: a list of nuscenes boxes
    :param radar_pc: a numpy array of the radar pointcloud (18,N) <really just (3,N)>
    :param lidar_pc: a numpy array of the lidar pointcloud (3,N)
    :param image: a numpy array of the rgb image
    """
    clf()
    if plot_origin:
        _plot_origin()
    
    if boxes is not None:
        corners = boxes3d_to_corners3d(boxes, swap=False, cam_coord=cam_coord)
        _plot_corners(corners)
    
    if radar_pc is not None:
        points3d(radar_pc[0, :], radar_pc[1, :], radar_pc[2, :], scale_factor=.5, color=(1,0,0))
   
    if lidar_pc is not None:
        points3d(lidar_pc[0,:], lidar_pc[1,:], lidar_pc[2,:], scale_factor=.1, color=(0,1,0))
    
    if image is not None:
        _show_image(image)
    
    if gt_boxes is not None:
        if isinstance(gt_boxes[0], Box):
            _plot_nusc_box_corners(gt_boxes)
        else:
            corners = boxes3d_to_corners3d(gt_boxes, swap=False, cam_coord=cam_coord)
            _plot_corners(corners, color=(1,1,0))
    

def show_proposals_from_pkl(path, frame_number=0):
    _plot_origin()
    print('loaing nuscenes')
    dataset = nsd.NuscenesDataset('../../data/datasets/nuscenes', split='train', nsweeps_lidar=5)

    boxes = dataset[frame_number]['annotations']
    _plot_nusc_box_corners(boxes)

    print('plotting lidar pointcloud')
    lidar_pc = dataset[frame_number]['lidar']['points'].points
    points3d(lidar_pc[0,:], lidar_pc[1,:], lidar_pc[2,:], scale_factor=.1, color=(0,1,0))
    
    print('lodaing pkl file')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    thing = data[frame_number].reshape((-1,7))
    corners = []
    for i in range(thing.shape[0]):
        corners.append(bbox_to_corners(thing[i,:]))

    print("LIDAR proposals this frame: %d"%len(corners))
    _plot_corners(corners)
    input('wait')

def show_boxList3d_and_data(boxlist3d, idx):
    nusc = nsd.NuscenesDataset('../data/datasets/nuscenes')
    image = nusc[int(idx/6)]['camera'][idx%6]['image']
    for box in boxlist3d.bbox:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0))
    _show_image(image)
    input()

def _plot_nusc_box_corners(box_list):
    corners = [box.corners() for box in box_list]
    _plot_corners(corners, color=(1,1,0))
    
def _plot_corners(corner_list, color=(0,0,1)):
    cornerOrder = [0,1,5,4,0,3,2,6,7,3,0,4,7,6,5,1,2]
    print('plotting boxes')
    for corners in tqdm(corner_list, desc="plotting corners"):
        corner = []
        for i in range(corners.shape[1]):
            corner.append([corners[0,i], corners[1,i], corners[2,i]])
        thisCorner = []
        for k in cornerOrder:
            thisCorner.append(corner[k])
        thisCorner = np.array(thisCorner)
        plot3d(thisCorner[:,0], thisCorner[:,1], thisCorner[:,2], np.ones(len(cornerOrder)), color=color)

def _plot_origin():
    print('plotting origin')
    points3d([0],[0],[0],scale_factor=2)

def _show_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('camera image', image)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
def show_image(image):
    _show_image(image)

def show_3dboxes_on_image(corners, img=None):
    if img is None:
        img = np.ones((900,1600,3), np.uint8)
    linewidth=2
    colors = ((0, 0, 255), (255, 0, 0), (155, 155, 155))
    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            cv2.line(img,
                    (int(prev[0]), int(prev[1])),
                    (int(corner[0]), int(corner[1])),
                    color, linewidth)
            prev = corner

    # Draw the sides
    for i in range(4):
        cv2.line(img,
                (int(corners.T[i][0]), int(corners.T[i][1])),
                (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                colors[2][::-1], linewidth)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], colors[0][::-1])
    draw_rect(corners.T[4:], colors[1][::-1])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    cv2.line(img,
            (int(center_bottom[0]), int(center_bottom[1])),
            (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
            colors[0][::-1], linewidth)
    # cv2.imshow('image', img)
    return img

def show_2d_box_on_image(box, img=None):
    if img is None:
        img = np.ones((900,1600,3), np.uint8)
    linewidth=2
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), thickness=3)
    # cv2.imshow('image', img)
    return img

if __name__ == "__main__":
    show_proposals_from_pkl('../../methods/pointRCNN_nusc/output/rpn/nusc_rpn/eval/epoch_165/train/out.pkl', frame_number=123)
