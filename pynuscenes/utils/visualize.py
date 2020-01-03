import os
import cv2
import time
import copy
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from pynuscenes.utils import constants as _C
from pynuscenes.utils.io_utils import save_fig
import pynuscenes.utils.nuscenes_utils as nsutils
from nuscenes.utils.geometry_utils import view_points, box_in_image


def visualize_sample_2d(sample, coordinates, out_path=None):
    """
    Visualize sample data from all sensors in 2D
    
    :param sample [dict]: sample dictionary returned from nuscenes_db
    :param coordinates [str]: sample data coordinate system: 'vehicle' or 'global'
    :return out_path: Path to save the figure at
    """

    ## Determine the grid size for cameras
    if len(sample['camera']) == 1:
        ## Only one image
        figure, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax = [ax]
    else:
        ## 6 images in two rows
        figure, ax = plt.subplots(2, 3, figsize=(16, 9))
        ax = ax.ravel()

    ## Plot pointclouds on image
    for i, cam in enumerate(sample['camera']):
        image = cam['image']
        
        ## Plot LIDAR data
        if len(sample['lidar']) > 0:
            lidar_pc = sample['lidar'][0]['pointcloud']
            lidar_pose_rec = sample['lidar'][0]['pose_record']
            lidar_pc, color = nsutils.map_pointcloud_to_image(lidar_pc,
                                            image, 
                                            cam['cs_record'],
                                            cam_pose_record=cam['pose_record'],
                                            pointsensor_pose_record=lidar_pose_rec,
                                            coordinates=coordinates)
            draw_points_on_image(image, lidar_pc, color, ax=ax[i], dot_size=2)

        ## Plot Radar data
        if len(sample['radar']) > 0:
            radar_pc = sample['radar'][0]['pointcloud']
            radar_pose_rec = sample['radar'][0]['pose_record']
            radar_pc, color = nsutils.map_pointcloud_to_image(radar_pc,
                                            image, 
                                            cam['cs_record'],
                                            cam_pose_record=cam['pose_record'],
                                            pointsensor_pose_record=radar_pose_rec,
                                            coordinates=coordinates,
                                            dot_size=15)
            draw_points_on_image(image, radar_pc, color, ax=ax[i], dot_size=18, edge_color=(1,1,1))
        
        ## Plot annotations on image
        cam_cs_rec = cam['cs_record']      
        for box in sample['anns']:
            if coordinates=='global':
                box = nsutils.global_to_vehicle(box, cam['pose_record'])
            box = nsutils.vehicle_to_sensor(box, cam_cs_rec)
            draw_gt_box_on_image(box, image, cam_cs_rec, ax[i])

    ## Display and save the figures
    if out_path is not None:
        save_fig('output.jpg',fig=figure, format='jpg')
    
    return figure
##------------------------------------------------------------------------------
def visualize_sample_3d(sample, coordinates, fig=None):
    """
    Visualize sample data from all sensors in 3D using mayavi
    
    :param sample [dict]: sample dictionary returned from nuscenes_db
    :param coordinates [str]: sample data coordinate system: 'vehicle' or 'global'
    :fig: An mayavi mlab figure object to display the 3D pointclouds on
    """
    import mayavi.mlab as mlab

    ## Create 3D figure
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, 
                          engine=None, size=(1600, 1000))
    mlab.clf(figure=fig)
    
    ## Draw LIDAR
    visualize_pointcloud_3d(sample['lidar'][0]['pointcloud'].points.T, 
                       fig=fig, 
                       pts_size=3,
                       scalar=sample['lidar'][0]['pointcloud'].points.T[:,2])
    ## Draw Radar
    visualize_pointcloud_3d(sample['radar'][0]['pointcloud'].points.T, 
                       fig=fig, 
                       pts_color=(1,0,0), 
                       pts_mode='sphere', 
                       pts_size=.5)

    ## Draw 3D annotation boxes
    visualize_boxes_3d(sample['anns'], fig)

    mlab.show(1)
    return fig
##------------------------------------------------------------------------------
def visualize_pointcloud_3d(pc, scalar=None, fig=None, bgcolor=(0,0,0), pts_size=4, 
            pts_mode='point', pts_color=None):
    """ 
    Draw lidar points
    :parma pc (nparray): numpy array (n,3) of XYZ
    :param scalar (list): 
    :param color (nparray): numpy array (n) of intensity or whatever
    :param fig: mayavi figure handler, if None create new one otherwise will use it
    :return fig: created or used fig
    """
    import mayavi.mlab as mlab

    if fig is None: 
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None,
                          engine=None, size=(1600, 1000))
    
    if pts_mode == 'point':
        if scalar is None:
            vis = mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color=pts_color, 
                        mode='point', colormap='gnuplot', figure=fig)
        else:
            vis = mlab.points3d(pc[:,0], pc[:,1], pc[:,2], scalar, 
                        mode='point', colormap='gnuplot', figure=fig, color=pts_color)
        vis.actor.property.set(representation='p', point_size=pts_size)
    else:
        mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color=pts_color, mode=pts_mode,
                      colormap = 'gnuplot', scale_factor=pts_size, figure=fig)
    
    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.6, figure=fig)
    
    return fig
##------------------------------------------------------------------------------
def visualize_boxes_3d(boxes, fig=None, bgcolor=(0,0,0), show_names=False):
    """ 
    Draw lidar points
    :parma boxes ([Box]): list of Box objects
    :param fig: mayavi figure handler
    :param bgcolor (r,g,b): background color
    :return fig: created or used fig
    """
    import mayavi.mlab as mlab

    if fig is None: 
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None,
                          engine=None, size=(1600, 1000))
    
    corner_list = []
    box_names = []
    for box in boxes:
        corner_list.append(np.array(box.corners()))
        box_names.append(box.name)
    corner_list = np.swapaxes(np.array(corner_list),1,2)
    draw_boxes_by_corner_3d(corner_list, box_names, fig=fig, draw_text=show_names, 
                            color=(0,0.85,0.1), line_width=3)
    # draw origin
    mlab.points3d(0, 0, 0, color=(0.0, 0.0, 0.8), mode='sphere', scale_factor=3, figure=fig)
    
    return fig
##------------------------------------------------------------------------------
def draw_points_on_image(image, points, colors, ax=None, dot_size=0.2, out_path=None, edge_color='face'):
    """
    Draw points on an image. Points must be already in image coordinates.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 16))

    ax.margins(x=0,y=0)
    ax.imshow(image)

    # colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
    ax.scatter(points[0, :], points[1, :], c=colors, s=dot_size, edgecolors=edge_color)
    ax.axis('off')

    if out_path is not None:
        save_fig(out_path, format='pdf')
##------------------------------------------------------------------------------
def draw_boxes_by_corner_3d(gt_boxes3d, box_names=None, fig=None, color=(1,1,1), 
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
    import mayavi.mlab as mlab

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
    mlab.view(azimuth=180, elevation=70, 
              focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], 
              distance=62.0, figure=fig)
    return fig
##------------------------------------------------------------------------------
def draw_gt_box_on_image(box, img, cam_cs_record, ax=None):
    """
    Show 3D boxes on the image. Boxes must be in camera's coordinate system
    :param boxes (Box): 3D boxes 
    :param img (ndarray<H,W,3>): image in BGR format
    :param cam_cs_record (dict): Camera cs_record
    :param ax (pyplot ax): Axes onto which to render
    """
    cam_intrinsic = np.array(cam_cs_record['camera_intrinsic'])
    if not box_in_image(box, cam_intrinsic, (1600, 900)):
        return
    
    ## Init axes
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 16))
    # Show image.
    ax.imshow(img)
    # Show boxes.
    box.render(ax, view=cam_intrinsic, normalize=True)

    # Limit visible range.
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis('off')
    ax.set_aspect('equal')
##------------------------------------------------------------------------------
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
##------------------------------------------------------------------------------
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
##------------------------------------------------------------------------------
# def render_cv2(im: np.ndarray,
#                 corners: np.ndarray,
#                 colors = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),
#                 linewidth: int = 2) -> None:
#     """
#     Renders box using OpenCV2.
#     :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
#     :param corners: 
#     :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
#     :param linewidth: Linewidth for plot.
#     """
#     def draw_rect(selected_corners, color):
#         prev = selected_corners[-1]
#         for corner in selected_corners:
#             cv2.line(im,
#                         (int(prev[0]), int(prev[1])),
#                         (int(corner[0]), int(corner[1])),
#                         color, linewidth)
#             prev = corner

#     # Draw the sides
#     for i in range(4):
#         cv2.line(im,
#                     (int(corners.T[i][0]), int(corners.T[i][1])),
#                     (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
#                     colors[2][::-1], linewidth)

#     # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
#     draw_rect(corners.T[:4], colors[0][::-1])
#     draw_rect(corners.T[4:], colors[1][::-1])

#     # Draw line indicating the front
#     center_bottom_forward = np.mean(corners.T[2:4], axis=0)
#     center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
#     cv2.line(im,
#                 (int(center_bottom[0]), int(center_bottom[1])),
#                 (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
#                 colors[0][::-1], linewidth)
#     return im
##------------------------------------------------------------------------------
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