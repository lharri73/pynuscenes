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


def render_sample_in_2d(sample, out_path=None):
    """
    Visualize sample data from all sensors in 2D
    
    :param sample [dict]: sample dictionary returned from nuscenes_db
    :param coordinates [str]: sample data coordinate system: 'vehicle' or 'global'
    :return out_path: Path to save the figure at
    """
    ## Birds Eye View x and y limits
    x_lim=(-30, 30)
    y_lim=(-30, 30)

    ## Determine the grid size for cameras
    if len(sample['camera']) == 1:
        figure1, ax1 = plt.subplots(1, 1, figsize=(16, 9))
        ax1 = [ax1]
    else:
        ## 6 images in two rows
        figure1, ax1 = plt.subplots(2, 3, figsize=(16, 9))
        ax1 = ax1.ravel()
    figure2, ax2 = plt.subplots(1, 1, figsize=(16, 9))

    ## Plot camera view
    for i, cam in enumerate(sample['camera']):
        image = cam['image']
        ax1[i].imshow(image)
        cam_intrinsic = np.array(cam['cs_record']['camera_intrinsic'])
        
        ## Plot LIDAR data
        if 'lidar' in sample:
            lidar_pc = sample['lidar']['pointcloud']
            render_pc_in_bev(lidar_pc, ax=ax2, point_size=2, x_lim=x_lim, y_lim=y_lim)
            
            lidar_pc_cam, depth = nsutils.map_pointcloud_to_camera(
                                            lidar_pc,
                                            cam['cs_record'],
                                            cam['pose_record'],
                                            sample['lidar']['pose_record'],
                                            coordinates=sample['coordinates'])
            # lidar_pc_cam = lidar_pc
            render_pc_in_image(lidar_pc_cam, image, cam_intrinsic, ax=ax1[i], point_size=2)
        
        ## Plot Radar data
        if 'radar' in sample:
            radar_pc = sample['radar']['pointcloud']
            render_pc_in_bev(radar_pc, ax=ax2, point_size=10, x_lim=x_lim, y_lim=y_lim)
            radar_pc_cam, depth = nsutils.map_pointcloud_to_camera(
                                            radar_pc,
                                            cam['cs_record'],
                                            cam['pose_record'],
                                            sample['radar']['pose_record'],
                                            coordinates=sample['coordinates'])
            render_pc_in_image(radar_pc_cam, image, cam_intrinsic, ax=ax1[i], point_size=10)
            
        ## Plot annotations on image
        for ann in sample['anns']:
            box = ann['box_3d']
            render_3dbox_in_bev([box], ax2, x_lim=x_lim, y_lim=y_lim)
            box = nsutils.map_annotation_to_camera(box, 
                                cam_cs_record = cam['cs_record'],
                                cam_pose_record = cam['pose_record'],
                                ref_pose_record = sample['ref_pose_record'],
                                coordinates = sample['coordinates'])
            render_3dbox_in_image(box, image, cam_intrinsic, ax1[i])
            if 'box_2d' in ann:
                draw_xywh_bbox(image, ann['box_2d'], ax1[i])

    ## Display and save the figures
    if out_path is not None:
        save_fig(out_path, fig=figure1, format='jpg')
        save_fig('bev_'+out_path, fig=figure2, format='jpg')
    
    return figure1
##------------------------------------------------------------------------------
def render_sample_in_3d(sample, coordinates, fig=None, show=False):
    """
    Visualize sample data from all sensors in 3D using mayavi
    
    :param sample [dict]: sample dictionary returned from nuscenes_db
    :param coordinates [str]: sample data coordinate system: 'vehicle' or 'global'
    :fig: An mayavi mlab figure object to display the 3D pointclouds on
    :show: If Ture, open an mayavi window and display. If False, just return figure
    """
    import mayavi.mlab as mlab

    ## Create 3D figure
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, 
                          engine=None, size=(1600, 1000))
    mlab.clf(figure=fig)
    
    ## Draw LIDAR
    render_pc_in_3d(sample['lidar']['pointcloud'].points.T, 
                       fig=fig, 
                       pts_size=3,
                       pts_color=(0.5,0.5,0.5),
                    #    scalar=sample['lidar']['pointcloud'].points.T[:,2],
                       )
    ## Draw Radar
    render_pc_in_3d(sample['radar']['pointcloud'].points.T, 
                       fig=fig, 
                       pts_color=(1,0,0), 
                       pts_mode='sphere', 
                       pts_size=.5)

    ## Draw 3D annotation boxes
    boxes = [s['box_3d'] for s in sample['anns']]
    render_3dbox_in_3d(boxes, fig)

    # mlab.view(azimuth=180, elevation=65, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)

    if show:
        mlab.show(1)

    return fig

##------------------------------------------------------------------------------
def render_pc_in_image(pc, image, camera_intrinsic, ax=None, point_size=1, edge_color='face'):
    """
    Render point clouds in image. Point cloud must be in the camera coordinate 
    system.

    :param pc (PointCloud): point cloud
    :param camera_intrinsic (np.array: 3, 3): camera intrinsics matrix
    :param ax (plt ax): Axes on which to render the points
    :param point_size (int): point size
    :edge_color (str): edge color for points
    """

    h, w, _= image.shape
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.margins(x=0,y=0)
    ax.imshow(image)

    ## Map points from camera coordinates to the image plane
    points, depths, mask = nsutils.map_pointcloud_to_image(pc, 
                                                           camera_intrinsic, 
                                                           img_shape=(w, h))
    ax.scatter(points[0, :], points[1, :], c=depths, s=point_size, edgecolors=edge_color)
    ax.axis('off')
    
    return ax
##------------------------------------------------------------------------------
def render_pc_in_bev(pc, ax=None, point_size=1, color='k', x_lim=(-20, 20), y_lim=(-20, 20)):
    """
    Render point clouds in Birds Eye View (BEV).
    pc can be in vehicle or point sensor coordinate system.

    :param pc (np.float32: m, n): point cloud as a numpy array
    :param ax (plt ax): Axes on which to render the points
    :param point_size (int): point size
    :param color: points color in Matplotlib color format
    :param x_lim (int, int): x (min, max) range for plotting
    :param y_lim (int, int): y (min, max) range for plotting
    """

    view = np.eye(4)    # bev view
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    points = view_points(pc[:3, :], view, normalize=False)
    ax.scatter(points[0, :], points[1, :], c=color, s=point_size)
    
    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='red')
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    return ax
##------------------------------------------------------------------------------
def render_3dboxes_in_image(boxes, img, cam_intrinsic, ax=None, colors=('b', 'r', 'k')):
    """
    Render 3D boxes on an image. Boxes must be in camera's coordinate system
    :param boxes (Box): 3D boxes 
    :param img (ndarray<H,W,3>): image
    :param cam_cs_record (dict): Camera cs_record
    :param ax (pyplot ax): Axes onto which to render
    """
    if ax is None:
        _, ax = plt.subplots()
    h, w, _ = img.shape
    ax.imshow(img)
    # ax.set_xlim(0, w)
    # ax.set_ylim(h, 0)
    # ax.axis('off')
    # ax.set_aspect('equal')

    for box in boxes:
        if not box_in_image(box, cam_intrinsic, (1600, 900)):
            continue
        box.render(ax, view=cam_intrinsic, normalize=True, linewidth=1, colors=colors)
    
    return ax
##------------------------------------------------------------------------------
def render_3dbox_in_bev(boxes, ax=None, x_lim=(-20, 20), y_lim=(-20, 20)):
    """
    Render 3D boxes in Birds Eye View (BEV).
    :param boxes (Box): List of 3D boxes 
    :param ax (pyplot ax): Axes onto which to render
    :param x_lim (int, int): x (min, max) range for plotting
    :param y_lim (int, int): y (min, max) range for plotting
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 16))

    view = np.eye(4)    # bev view
    for box in boxes:
        box.render(ax, view=view, normalize=False)

    # Limit visible range.
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    # ax.axis('off')
    
    return ax
##------------------------------------------------------------------------------
def render_pc_in_3d(pc, scalar=None, fig=None, bgcolor=(0,0,0), pts_size=4, 
            pts_mode='point', pts_color=None, show_origin=False):
    """ 
    Draw lidar points
    :param pc (nparray): numpy array (n,3) of XYZ
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
    
    ## draw origin
    if show_origin:
        mlab.points3d(0, 0, 0, color=(0.0, 0.0, 0.8), mode='sphere', scale_factor=3, figure=fig)    
    
    return fig
##------------------------------------------------------------------------------
def render_3dbox_in_3d(boxes, fig=None, bgcolor=(0,0,0), show_names=False, show_origin=False):
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
    
    ## draw origin
    if show_origin:
        mlab.points3d(0, 0, 0, color=(0.0, 0.0, 0.8), mode='sphere', scale_factor=3, figure=fig)
    
    return fig
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
def show_3dBoxes_on_image(boxes, img, cam_cs_record):
    """ ## TODO: Check compatibility
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
def render_2dbox_in_image(bbox, image, out_dir=None, img_id=None):
    """ ## TODO: Check compatibility
    Show 2D boxes in [xyxy] format on the image
    :param bbox (list): list of 2D boxes in [xyxy] format
    :param img (ndarray<H,W,3>): image in BGR format
    
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i, this_box_corners in enumerate(bbox):
        img = copy.deepcopy(image)
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
def draw_xywh_bbox(box, ax=None, color=(0,255,0), lineWidth=3, format='BGR'):
    
    import matplotlib.patches as patches
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 9))
        
    rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=lineWidth,
                                edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    return ax

    # assert format in ['RGB', 'BGR'], "Format must be either 'BGR' or 'RGB'."
    
    # img = copy.deepcopy(img)
    # img = np.asarray(img)
    # if format == 'RGB':
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # box = [int(elem) for elem in box]
    # cv2.rectangle(img,(box[0],box[1]), (box[0]+box[2], box[1]+box[3]),
    #                 color,lineWidth)

    # if format == 'RGB':
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # cv2.imwrite('2d_box.jpg', img)
    # return img
##------------------------------------------------------------------------------
def arrange_images_PIL(image_list: list, 
                       im_size: tuple=(640,360),
                       grid_size: tuple=(2,2)) -> np.ndarray:
    """ ## TODO: Check compatibility
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