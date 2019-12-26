"""
# Code based on export_kitti.py written by Holger Caesar, 2019.
"""
import os
import json
from typing import List, Dict, Any
from tqdm import tqdm

from pyquaternion import Quaternion
import numpy as np
import fire
from PIL import Image
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud, Box
from nuscenes.utils.splits import create_splits_logs
from nuscenes.utils.kitti import KittiDB
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.geometry_utils import view_points
from pynuscenes.utils import constants as _C


class KittiConverter:
    def __init__(self,
                 output_dir: str = '../../data/nusc_kitti',
                 nusc_dir: str = '../../data/datasets/nuscenes',
                 nusc_version: str = 'v1.0-mini',
                 split: str = 'mini_train',
                 cam_name: str = 'CAM_FRONT',
                 lidar_name: str = 'LIDAR_TOP',
                 lidar_sweeps: int=10,
                 radar_sweeps: int=1,
                 image_count: int = None,
                 use_symlinks: bool = False):
        """
        :param output_dir: Where to write the KITTI-style annotations.
        :param cam_name: Name of the camera to export.
        :param lidar_name: Name of the lidar sensor.
        :param lidar_sweeps: Number of lidar sweeps
        :param radar_sweeps: Number of radar sweeps
        :param image_count: Number of images to convert.
        :param nusc_version: nuScenes version to use.
        :param split: Dataset split to use.
        """

        self.output_dir = os.path.expanduser(output_dir)
        self.cam_name = cam_name
        self.lidar_name = lidar_name
        self.lidar_sweeps = lidar_sweeps
        self.radar_sweeps = radar_sweeps
        self.image_count = image_count
        self.nusc_version = nusc_version
        self.split = split
        self.nusc_dir = nusc_dir
        self.use_symlinks = use_symlinks

        # Create output_dir.
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # Select subset of the data to look at.
        self.nusc = NuScenes(version=nusc_version,
                             dataroot=nusc_dir)

    def nuscenes_gt_to_kitti(self) -> None:
        """
        Converts nuScenes GT annotations to KITTI format.
        """
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        imsize = (1600, 900)

        token_idx = 0  # Start tokens from 0.

        # Create output folders.
        label_folder = os.path.join(self.output_dir, self.split, 'label_2')
        calib_folder = os.path.join(self.output_dir, self.split, 'calib')
        image_folder = os.path.join(self.output_dir, self.split, 'image_2')
        lidar_folder = os.path.join(self.output_dir, self.split, 'velodyne')
        radar_folder = os.path.join(self.output_dir, self.split, 'radar')
        for folder in [label_folder, calib_folder, image_folder, lidar_folder, radar_folder]:
            if not os.path.isdir(folder):
                os.makedirs(folder)
        id_to_token_file = os.path.join(self.output_dir, self.split, 'id2token.txt')
        id2token = open(id_to_token_file, "w+")

        # Use only the samples from the current split.
        split_logs = create_splits_logs(self.split, self.nusc)
        sample_tokens = self._split_to_samples(split_logs)
        sample_tokens = sample_tokens[:self.image_count]

        out_filenames = []
        for sample_token in tqdm(sample_tokens):
            # Get sample data.
            sample = self.nusc.get('sample', sample_token)
            sample_annotation_tokens = sample['anns']
            cam_token = sample['data'][self.cam_name]
            lidar_token = sample['data'][self.lidar_name]
            radar_tokens = []
            for radar_name in _C.RADARS.keys():
                radar_tokens.append(sample['data'][radar_name])
                
            # Retrieve sensor records.
            sd_record_cam = self.nusc.get('sample_data', cam_token)
            sd_record_lid = self.nusc.get('sample_data', lidar_token)
            cs_record_cam = self.nusc.get('calibrated_sensor', 
                                        sd_record_cam['calibrated_sensor_token'])
            cs_record_lid = self.nusc.get('calibrated_sensor', 
                                        sd_record_lid['calibrated_sensor_token'])
            sd_record_rad = []
            cs_record_rad = []
            for i, radar_token in enumerate(radar_tokens):
                sd_record_rad.append(self.nusc.get('sample_data', radar_token))
                cs_record_rad.append(self.nusc.get('calibrated_sensor', 
                                    sd_record_rad[i]['calibrated_sensor_token']))

            # Combine transformations and convert to KITTI format.
            # Note: cam uses same conventions in KITTI and nuScenes.
            lid_to_ego = transform_matrix(cs_record_lid['translation'], 
                                          Quaternion(cs_record_lid['rotation']),
                                          inverse=False)
            ego_to_cam = transform_matrix(cs_record_cam['translation'], 
                                          Quaternion(cs_record_cam['rotation']),
                                          inverse=True)
            rad_to_ego = []
            for cs_rec_rad in cs_record_rad:
                rad_to_ego.append(transform_matrix(cs_rec_rad['translation'], 
                                          Quaternion(cs_rec_rad['rotation']),
                                          inverse=False))

            velo_to_cam = np.dot(ego_to_cam, lid_to_ego)
            # # TODO: Assuming Radar point are going to be in ego coordinates
            # radar_to_cam = ego_to_cam

            # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
            velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)
            # # Nuscenes radars use same convention as KITTI lidar
            # radar_to_cam_kitti = radar_to_cam

            # Currently not used.
            imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
            r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

            # Projection matrix.
            p_left_kitti = np.zeros((3, 4))
            # Cameras are always rectified.
            p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']

            # Create KITTI style transforms.
            velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
            velo_to_cam_trans = velo_to_cam_kitti[:3, 3]
            # radar_to_cam_rot = radar_to_cam_kitti[:3, :3]
            # radar_to_cam_trans = radar_to_cam_kitti[:3, 3]

            # Check that the lidar rotation has the same format as in KITTI.
            assert (velo_to_cam_rot.round(0) == np.array([[0, -1, 0], 
                                                          [0, 0, -1], 
                                                          [1, 0, 0]])).all()
            assert (velo_to_cam_trans[1:3] < 0).all()

            # Retrieve the token from the lidar.
            # Note that this may be confusing as the filename of the camera will 
            # include the timestamp of the lidar, not the camera.
            filename_cam_full = sd_record_cam['filename']
            filename_lid_full = sd_record_lid['filename']
            filename_rad_full=[]
            for sd_rec_rad in sd_record_rad:
                filename_rad_full.append(sd_rec_rad['filename'])
            out_filename = '%06d' % token_idx # Alternative to use KITTI names.
            # out_filename = sample_token
            
            # Write token to disk.
            text = sample_token
            id2token.write(text + '\n')
            id2token.flush()
            token_idx += 1

            # Convert image (jpg to png).
            src_im_path = os.path.join(self.nusc.dataroot, filename_cam_full)
            dst_im_path = os.path.join(image_folder, out_filename + '.png')
            
            if self.use_symlinks:
                # Create symbolic links to nuscenes images
                os.symlink(os.path.abspath(src_im_path), dst_im_path)
            else:
                im = Image.open(src_im_path)
                im.save(dst_im_path, "PNG")

            # Convert lidar.
            src_lid_path = os.path.join(self.nusc.dataroot, filename_lid_full)
            dst_lid_path = os.path.join(lidar_folder, out_filename + '.bin')
            assert not dst_lid_path.endswith('.pcd.bin')
            # pcl = LidarPointCloud.from_file(src_lid_path)
            pcl, _ = LidarPointCloud.from_file_multisweep(
                             nusc=self.nusc,
                             sample_rec=sample,
                             chan=self.lidar_name,
                             ref_chan=self.lidar_name,
                             nsweeps=self.lidar_sweeps,
                             min_distance=1)
            
            pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)  # In KITTI lidar frame.
            pcl.points = pcl.points.astype('float32')
            with open(dst_lid_path, "w") as lid_file:
                pcl.points.T.tofile(lid_file)
            
            # # Visualize pointclouds
            # _, ax = plt.subplots(1, 1, figsize=(9, 9))
            # points = view_points(pcl.points[:3, :], np.eye(4), normalize=False)
            # dists = np.sqrt(np.sum(pcl.points[:2, :] ** 2, axis=0))
            # colors = np.minimum(1, dists / 50)
            # ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)
            # # plt.show(block=False)
            # plt.show()
            
            # Convert radar.
            src_rad_path = []
            for filename_radar in filename_rad_full:
                src_rad_path.append(os.path.join(self.nusc.dataroot, filename_radar))
            dst_rad_path = os.path.join(radar_folder, out_filename + '.bin')
            assert not dst_rad_path.endswith('.pcd.bin')
            pcl = RadarPointCloud(np.zeros((18, 0)))
            ## Get Radar points in Lidar coordinate system
            for i,rad_path in enumerate(src_rad_path):
                pc, _ = RadarPointCloud.from_file_multisweep(self.nusc,
                                            sample_rec=sample, 
                                            chan=sd_record_rad[i]['channel'], 
                                            ref_chan=self.lidar_name,
                                            nsweeps=self.radar_sweeps,
                                            min_distance=0)

                # rot_matrix = Quaternion(cs_record_rad[i]['rotation']).rotation_matrix
                # pc.rotate(rot_matrix)
                # pc.translate(np.array(cs_record_rad[i]['translation']))
                pcl.points = np.hstack((pcl.points, pc.points))
            pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)  # In KITTI lidar frame.
            ## Visualize pointclouds
            # _, ax = plt.subplots(1, 1, figsize=(9, 9))
            # points = view_points(pcl.points[:3, :], np.eye(4), normalize=False)
            # dists = np.sqrt(np.sum(pcl.points[:2, :] ** 2, axis=0))
            # colors = np.minimum(1, dists / 50)
            # ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)
            # plt.show()
            
            pcl.points = pcl.points.astype('float32')
            with open(dst_rad_path, "w") as rad_file:
                pcl.points.T.tofile(rad_file)

            # Add to tokens.
            out_filenames.append(out_filename)

            # Create calibration file.
            kitti_transforms = dict()
            kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['P2'] = p_left_kitti  # Left camera transform.
            kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['R0_rect'] = r0_rect.rotation_matrix  # Cameras are already rectified.
            kitti_transforms['Tr_velo_to_cam'] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
            # kitti_transforms['Tr_radar_to_cam'] = np.hstack((radar_to_cam_rot, radar_to_cam_trans.reshape(3, 1)))
            kitti_transforms['Tr_imu_to_velo'] = imu_to_velo_kitti
            calib_path = os.path.join(calib_folder, out_filename + '.txt')
            with open(calib_path, "w") as calib_file:
                for (key, val) in kitti_transforms.items():
                    val = val.flatten()
                    val_str = '%.12e' % val[0]
                    for v in val[1:]:
                        val_str += ' %.12e' % v
                    calib_file.write('%s: %s\n' % (key, val_str))

            # Write label file.
            label_path = os.path.join(label_folder, out_filename + '.txt')
            if os.path.exists(label_path):
                print('Skipping existing file: %s' % label_path)
                continue
            with open(label_path, "w") as label_file:
                for sample_annotation_token in sample_annotation_tokens:
                    sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)

                    # Get box in LIDAR frame.
                    _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                                     selected_anntokens=[sample_annotation_token])
                    box_lidar_nusc = box_lidar_nusc[0]

                    # Truncated: Set all objects to 0 which means untruncated.
                    truncated = 0.0

                    # Occluded: Set all objects to full visibility as this information is not available in nuScenes.
                    occluded = 0

                    # Convert nuScenes category to nuScenes detection challenge category.
                    detection_name = _C.KITTI_CLASSES.get(sample_annotation['category_name'])
                    # Skip categories that are not in the KITTI classes.
                    if detection_name is None:
                        continue

                    # Convert from nuScenes to KITTI box format.
                    box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                        box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

                    # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                    bbox_2d = KittiDB.project_kitti_box_to_image(box_cam_kitti, p_left_kitti, imsize=imsize)
                    if bbox_2d is None:
                        # continue
                        ## If box is not inside the image, 2D boxes are set to zero
                        bbox_2d = (0,0,0,0)

                    # Set dummy score so we can use this file as result.
                    box_cam_kitti.score = 0

                    # Convert box to output string format.
                    output = KittiDB.box_to_string(name=detection_name, box=box_cam_kitti, bbox_2d=bbox_2d,
                                                   truncation=truncated, occlusion=occluded)

                    # Write to disk.
                    label_file.write(output + '\n')
        id2token.close()


    def render_kitti(self, render_2d: bool) -> None:
        """
        Renders the annotations in the KITTI dataset from a lidar and a camera view.
        :param render_2d: Whether to render 2d boxes (only works for camera data).
        """
        if render_2d:
            print('Rendering 2d boxes from KITTI format')
        else:
            print('Rendering 3d boxes projected from 3d KITTI format')

        # Load the KITTI dataset.
        kitti = KittiDB(root=self.output_dir, splits=(self.split,))

        # Create output folder.
        render_dir = os.path.join(self.output_dir, 'render')
        if not os.path.isdir(render_dir):
            os.mkdir(render_dir)

        # Render each image.
        for token in kitti.tokens[:self.image_count]:
            print(token)
            input('here')
            for sensor in ['lidar', 'radar', 'camera']:
                out_path = os.path.join(render_dir, '%s_%s.png' % (token, sensor))
                print('Rendering file to disk: %s' % out_path)
                kitti.render_sample_data(token, sensor_modality=sensor, out_path=out_path, render_2d=render_2d)
                plt.close()  # Close the windows to avoid a warning of too many open windows.


    def kitti_res_to_nuscenes(self, meta: Dict[str, bool] = None) -> None:
        """
        Converts a KITTI detection result to the nuScenes detection results format.
        :param meta: Meta data describing the method used to generate the result. See nuscenes.org/object-detection.
        """
        # Dummy meta data, please adjust accordingly.
        if meta is None:
            meta = {
                'use_camera': False,
                'use_lidar': True,
                'use_radar': False,
                'use_map': False,
                'use_external': False}
        # Init.
        results = {}

        # Load the KITTI dataset.
        kitti = KittiDB(root=self.output_dir, splits=(self.split, ))

        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(self.split, self.nusc)

        # Use only the samples from the current split.
        sample_tokens = self._split_to_samples(split_logs)
        sample_tokens = sample_tokens[:self.image_count]

        for sample_token in sample_tokens:
            # Get the KITTI boxes we just generated in LIDAR frame.
            kitti_token = '%s_%s' % (self.split, sample_token)
            boxes = kitti.get_boxes(token=kitti_token)

            # Convert KITTI boxes to nuScenes detection challenge result format.
            sample_results = [self._box_to_sample_result(sample_token, box) for box in boxes]

            # Store all results for this image.
            results[sample_token] = sample_results

        # Store submission file to disk.
        submission = {
            'meta': meta,
            'results': results
        }
        submission_path = os.path.join(self.output_dir, 'submission.json')
        print('Writing submission to: %s' % submission_path)
        with open(submission_path, 'w') as f:
            json.dump(submission, f, indent=2)


    def _box_to_sample_result(self, sample_token: str, box: Box, attribute_name: str = '') -> Dict[str, Any]:
        # Prepare data
        translation = box.center
        size = box.wlh
        rotation = box.orientation.q
        velocity = box.velocity
        detection_name = box.name
        detection_score = box.score

        # Create result dict
        sample_result = dict()
        sample_result['sample_token'] = sample_token
        sample_result['translation'] = translation.tolist()
        sample_result['size'] = size.tolist()
        sample_result['rotation'] = rotation.tolist()
        sample_result['velocity'] = velocity.tolist()[:2]  # Only need vx, vy.
        sample_result['detection_name'] = detection_name
        sample_result['detection_score'] = detection_score
        sample_result['attribute_name'] = attribute_name

        return sample_result


    def _split_to_samples(self, split_logs: List[str]) -> List[str]:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        samples = []
        for sample in self.nusc.sample:
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            logfile = log['logfile']
            if logfile in split_logs:
                samples.append(sample['token'])
        return samples


if __name__ == '__main__':
    converter = KittiConverter()    
    converter.nuscenes_gt_to_kitti()