Usage and Format
================

Usage
-----

To instantiate an instance of the dataset, use the following:

.. code-block:: python

    from pynuscenes.nuscenes_dataset import NuscenesDataset

    nusc = NuscenesDataset(dataroot="/data/datasets/nuscenes",
                           version="v1.0-mini",
                           split="mini_train",
                           cfg="pynuscenes/config/cfg.yml")


The dataset can then be iterated using standard iterators and brace notation:

.. code-block:: python

    for sample in nusc:
        print(sample)

or

.. code-block:: python

   sample = nusc[0]
   print(sample)


Iterator Format
---------------
The returned "sample" from the above examples is a dictionary that uses the following schema:

.. code-block:: python

   frame = {
       'camera': [{
           image: np.array,                      # Image from this camera
           cs_record: dict,                      # Camera sensor calibration parameters
           pose_record: dict,                    # Vehicle pose record for the timestamp of the camera
           image_id: int,
           token: str,
           filename: str,
           channel: str,
           width: int,
           height: int,
           }, ...
       ],
       'lidar': {
           pointcloud: nuscenes.LidarPointCloud, # LIDAR Pointcloud (raw points are at ['pointcloud'].points (4 x n))
           pose_record: dict,                    # Vehicle pose record for the timestamp of the lidar
           token: str,
           filename: str,
           channel: str,
       },
       'radar': {
           pointcloud: nuscenes.RadarPointCloud, # Radar Pointcloud (raw points are at ['pointcloud'].points (18 x n))
           pose_record: dict,                    # Vehicle pose record for the timestamp of Radar
       },
       'anns': [{},...]                          # Filtered annotations as Box objects
       'ref_pose_record': {},                    # Reference pose record used for mapping anns from global to vehicle
       'sample_token': str,
       'coordinates': str,
       'meta': dict,
       'id': int
   }

The ["lidar"]["pointcloud"] field in the above structure is an instance of the `nuscenes.LidarPointCloud <https://github.com/nutonomy/nuscenes-devkit/blob/d9c36a603898965fc2c8111f4bdf4ed1f8f10aa7/python-sdk/nuscenes/utils/data_classes.py#L236>`_ class.

The ["radar"]["pointcloud"] field in the above structure is a single instance of the `nuscenes.RadarPointCloud <https://github.com/nutonomy/nuscenes-devkit/blob/d9c36a603898965fc2c8111f4bdf4ed1f8f10aa7/python-sdk/nuscenes/utils/data_classes.py#L261>`_ class
that contains points from *all* radar sensors enabled in the configuration file.

The ["anns"] field in the above structure is a list of dictionaries that use the following schema:

.. code-block:: python

   ann = {
       'category_id': str,                       # Category token
       'num_lidar_pts': str,                     # Number of lidar points in this box
       'num_radar_pts': str,                     # Number of radar points in this box
       'instance_token': str,                    # Which object instance is this annotating. An instance can have multiple annotations over time.
       'distance': float,                        # Distance from the reference sensor (meters)
       'box_3d': nuscenes.utils.data_classes.Box # Instance of the nuscenes box describing this object
   }

The ['box_3d'] field in the above structure is an instance of the `nuscenes.utils.data_classes.Box <https://github.com/nutonomy/nuscenes-devkit/blob/d9c36a603898965fc2c8111f4bdf4ed1f8f10aa7/python-sdk/nuscenes/utils/data_classes.py#L521>`_ class.

