import cv2
from pynuscenes.utils import constants as _C
import numpy as np
import pynuscenes.utils.nuscenes_utils as nsutils
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import copy
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
import time


def show_sample_data(sample, coordinates='vehicle', fig=None):
    """
    Render the data from all sensors in a single sample
    
    :param sample [dict]: sample dictionary returned from nuscenes_db
    :param coordinates [str]: sample data coordinate system: 'vehicle' or 'global'
    :fig: An mayavi mlab figure object to display the 3D pointclouds on
    """

    cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
               'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    image_list = []

    for i, cam in enumerate(sample['camera']):
        image = cam['image']
        image = map_pointcloud_to_image(sample['lidar'][0]['pointcloud'],
                image, 
                cam['cs_record'],
                coordinates=coordinates,
                ego_pose=sample['ego_pose'])
        ## Map RADAR points
        image = map_pointcloud_to_image(sample['radar'][0]['pointcloud'],
                image, 
                cam['cs_record'],
                radius=8,
                coordinates=coordinates,
                ego_pose=sample['ego_pose'])
        image_list.append(image)

    ## Draw 3D point clouds
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, 
                          engine=None, size=(1600, 1000))
    mlab.clf(figure=fig)
    draw_pc(sample['lidar'][0]['pointcloud'].points.T, fig=fig, pts_size=3,
            scalar=sample['lidar'][0]['pointcloud'].points.T[:,2])
    draw_pc(sample['radar'][0]['pointcloud'].points.T, fig=fig, pts_color=(1,0,0), 
            pts_mode='sphere', pts_size=.5)

    ## Draw 3D annotation boxes
    corner_list = []
    box_names = []
    for box in sample['anns']:
        corner_list.append(np.array(box.corners()))
        box_names.append(box.name)
    corner_list = np.swapaxes(np.array(corner_list),1,2)
    draw_gt_boxes3d(corner_list, box_names, fig=fig, draw_text=False, 
                    color=(0,0.85,0.1), line_width=3)
    
    image = arrange_images_PIL(image_list, grid_size=(2,3))
    mlab.show(1)
    return fig
##--------------------------------------------------------------------------
def arrange_images_PIL(image_list: list, 
                       im_size: tuple=(640,360),
                       grid_size: tuple=(2,2)) -> np.ndarray:
    """
    Arranges multiple images into a single image
    
    :param image_list: list of images
    :param im_size: Size of each image in the grid (width, height)
    :param grid_size: grid size
    :return: grid image with all images
    """
    assert len(image_list) <= grid_size[0]*grid_size[1], \
        "Provided more images than grid size."
    
    nrow = grid_size[0]
    ncol = grid_size[1]
    width = im_size[0]
    height = im_size[1]
    cvs = Image.new('RGB',(width*ncol, height*nrow))

    for i, image in enumerate(image_list):
        pil_image = Image.fromarray(image).resize(im_size)
        px, py = width*(i%ncol), height*int(i/ncol)
        cvs.paste(pil_image,(px,py))

    cvs.show()
    return cvs
##--------------------------------------------------------------------------
def _arrange_images(image_list: list, im_size: tuple=(640,360)) -> np.ndarray:
    """
    Arranges multiple images into a single image
    :param image_list: list rows where a row is a list of images
    :param image_size: new size of the images
    :return: the new image as a numpy array
    """
    rows = []
    for row in image_list:
        rows.append(np.hstack((cv2.cvtColor(cv2.resize(pic, im_size), 
                    cv2.COLOR_RGB2BGR)) for pic in row))
    image = np.vstack((row) for row in rows)
    return image
##--------------------------------------------------------------------------
def map_pointcloud_to_image(pc, im, cam_cs_record, coordinates='vehicle', 
                            radius=2, ego_pose=None):
    """
    Given a point sensor (lidar/radar) point cloud and camera image, 
    map the point cloud to the image.
    
    :param pc: point cloud
    :param im: Camera image.
    :param cam_cs_record: Camera calibrated sensor record
    :param coordinates [str]: Point cloud coordinates ('vehicle', 'global') 
    :param radius: Radius of each plotted point in the point cloud
    :param ego_pose: Vehicle's ego-pose if points are in 'global' coordinates
    :return image: Camera image with mapped point cloud.
    """
    ## Transform into the camera.
    pc = nsutils.pc_to_sensor(pc, cam_cs_record, coordinates=coordinates, 
                                      ego_pose=ego_pose)

    ## Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    coloring = depths

    ## Take the actual picture
    points = view_points(pc.points[:3, :], 
                         np.array(cam_cs_record['camera_intrinsic']), 
                         normalize=True)

    ## Remove points that are either outside or behind the camera. 
    # Leave a margin of 1 pixel for aesthetic reasons.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.shape[0] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    im = plot_points_on_image(im, points.T, coloring, radius)

    # plt.figure(figsize=(9, 16))
    #     plt.imshow(im)
    #     plt.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
    #     plt.axis('off')

    return im
##--------------------------------------------------------------------------
def plot_points_on_image(image, points, coloring, radius):
    newPoint = [0,0]
    coloring = coloring * 255.0 / 20.0
    for i, point in enumerate(points):
        newPoint[0], newPoint[1] = int(point[0]), int(point[1])
        cv2.circle(image, tuple(newPoint), radius, 
                   (int(coloring[i]),0,255-int(coloring[i])), -1)
    return image
##--------------------------------------------------------------------------
def draw_gt_boxes3d(gt_boxes3d, box_names=None, fig=None, color=(1,1,1), 
                    line_width=1, draw_text=True, text_scale=(.5,.5,.5), 
                    color_list=None):
    '''
    Draw 3D bounding boxes
    :param gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
    :param box_names: list of names for every box
    :param fig: mayavi figure handler
    :param color: RGB value tuple in range (0,1), box line color
    :param line_width: box line width
    :param draw_text: boolean, if true, write box indices beside boxes
    :param text_scale: three number tuple
    :param color_list: a list of RGB tuple, if not None, overwrite color.
    :return fig: updated fig
    ''' 
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, 
                          engine=None, size=(1600, 1000))
    if box_names is None:
        box_names = []
        for i in range(gt_boxes3d.shape[0]):
            box_names.append(str(i))
    for n, name in enumerate(box_names):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n] 
        if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%s'%name, 
                                  scale=text_scale, color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], 
                        color=color, tube_radius=None, line_width=line_width, 
                        figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], 
                        color=color, tube_radius=None, line_width=line_width, 
                        figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], 
                        color=color, tube_radius=None, line_width=line_width, 
                        figure=fig)
    #mlab.show(1)
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig
##--------------------------------------------------------------------------
def draw_pc(pc, scalar=None, fig=None, bgcolor=(0,0,0), pts_size=4, 
            pts_mode='point', pts_color=None):
    """ 
    Draw lidar points
    :parma pc: numpy array (n,3) of XYZ
    :param color: numpy array (n) of intensity or whatever
    :param fig: mayavi figure handler, if None create new one otherwise will use it
    :return fig: created or used fig
    """
    if fig is None: 
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None,
                          engine=None, size=(1600, 1000))

    if pts_mode == 'point':
        if scalar is None:
            vis = mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color=pts_color, 
                        mode='point', colormap='gnuplot', figure=fig)
        else:
            vis = mlab.points3d(pc[:,0], pc[:,1], pc[:,2], scalar, 
                        mode='point', colormap='gnuplot', figure=fig)
        vis.actor.property.set(representation='p', point_size=pts_size)
    
    else:
        mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color=pts_color, mode=pts_mode,
                      colormap = 'gnuplot', scale_factor=pts_size, figure=fig)
    
    return fig

##--------------------------------------------------------------------------
def show_3dBoxes_on_image(boxes, img, cam_cs_record):
    """
    Show 3D boxes in [x,y,z,w,l,h,ry] format on the image
    :param boxes (ndarray<N,7>): 3D boxes 
    :param img (ndarray<H,W,3>): image in BGR format
    :param cam_cs_record (dict): nuscenes calibration record
    
    """
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    corners_3d = nsutils.bbox_to_corners(boxes)
    corners_2d, _ = nsutils.corners3d_to_image(corners_3d, cam_cs_record, (img.shape[1], 
                                     img.shape[0]))
    for this_box_corners in corners_2d:
        img = render_cv2(img, this_box_corners)
        cv2.imshow('image', img)
        cv2.waitKey(1)

##--------------------------------------------------------------------------
def show_2dBoxes_on_image(img_corners_2d, image, 
                          img_corners_3d=None,
                          out_dir=None,
                          img_id=None):
    """
    Show 2D boxes in [xyxy] format on the image
    :param img_corners_2d (list): list of 2D boxes 
    :param img (ndarray<H,W,3>): image in BGR format
    :param img_corners_3d (ndarray<N,7>): Optional 3D boxes
    
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i, this_box_corners in enumerate(img_corners_2d):
        img = copy.deepcopy(image)
        if img_corners_3d is not None:
            img = render_cv2(img, img_corners_3d[i])
        cv2.rectangle(img, (int(this_box_corners[0]), int(this_box_corners[1])), 
                        (int(this_box_corners[2]), int(this_box_corners[3])), 
                        (0,255,0), 2)
        cv2.imshow('image', img)
        # cv2.waitKey(1)
        k = cv2.waitKey(0)
        if  k==32:    # Space key to continue
            continue
        elif k == ord('s'):
            out_file = os.path.join(out_dir, str(img_id)+'_'+str(i)+'.jpg')
            cv2.imwrite(out_file, img)
            continue

##--------------------------------------------------------------------------
def render_cv2(im: np.ndarray,
                corners: np.ndarray,
                colors = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),
                linewidth: int = 2) -> None:
    """
    Renders box using OpenCV2.
    :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
    :param corners: 
    :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
    :param linewidth: Linewidth for plot.
    """
    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            cv2.line(im,
                        (int(prev[0]), int(prev[1])),
                        (int(corner[0]), int(corner[1])),
                        color, linewidth)
            prev = corner

    # Draw the sides
    for i in range(4):
        cv2.line(im,
                    (int(corners.T[i][0]), int(corners.T[i][1])),
                    (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                    colors[2][::-1], linewidth)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], colors[0][::-1])
    draw_rect(corners.T[4:], colors[1][::-1])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    cv2.line(im,
                (int(center_bottom[0]), int(center_bottom[1])),
                (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                colors[0][::-1], linewidth)
    return im