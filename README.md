# PyNuscnes

PyNuscenes is a dataloader for the [NuScenes](https://www.nuscenes.org/) dataset.
It uses the [NuScenes devkit](https://github.com/nutonomy/nuscenes-devkit) and 
provides APIs for loading sensor data in different coordinate systems.

## Features:
    - Return sensor data in vehicle or global coorinate systems
    - Return different camera images individually or altogether as one sample.
    - Include sweeps for Radar and LiDAR data
    - Convert the dataset to Kitti and COCO formats
    - Visualize sample data in 2D and 3D environments

## Installation
### Requirements
- Linux or macOS
- Python>= 3.6
- pycocotools: 
    ```bash
    pip install cython pycocotools
    ```
- cocoplus:
    ```bash
    git clone https://github.com/lharri73/cocoapi_plus.git
    cd cocoapi_plus
    pip install -e .
    ```

### Build nuscenes_dataset
After having the above dependencies, run:
```bash
git clone https://github.com/lharri73/nuscenes_dataset.git
cd nuscenes_dataset
pip install -e .
```

## Getting Started

#### Frame Structure

Dataloader frames returned from the iterator
```yaml
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
        pointcloud: nuscenes.LidarPointCloud, # LIDAR Pointcloud (raw points are at ['pointcloud'].points 4xn)
        pose_record: dict,                    # Vehicle pose record for the timestamp of the lidar
        token: str,
        filename: str,
        channel: str,
    },
    'radar': {
        pointcloud: nuscenes.RadarPointCloud, # Radar Pointcloud (raw points are at ['pointcloud'].points 18xn)
        pose_record: dict,                    # Vehicle pose record for the timestamp of Radar
    },
    'anns': [{},...]                          # Filtered annotations as Box objects
    'ref_pose_record': {},                    # Reference pose record used for mapping anns from global to vehicle
    'sample_token': str,
    'coordinates': str,
    'meta': dict,
    'id': int
}
```

Database frame (mostly used internally):
```yaml
frame = {
    'camera': [{           # A list of camera frame dictionaries (One for each camera)
        token: str,        # Camera sensor record token
        filename: str,     # Image filename, relative to nuscenes root dir
        cs_record: dict,   # Camera sensor calibration parameters
        pose_record: dict, # Vehicle pose record for the timestamp of the camera
        channel: str,      # Camera channel (e.g. CAM_FRONT_RIGHT)
        width: int,        # Image width
        height: int,       # Image height
        image_id: int,     # Image ID
        }, ...
    ],
    'lidar': {             # A LIDAR frame dictionaries
        token: str,        # LIDAR sensor record token
        filename: str,     # Pointcloud filename, relative to nuscenes root dir
        channel: str,      # LIDAR channel (always LIDAR_TOP)
        sc_record: dict,   # LIDAR sensor calibration parameters
        pose_record: dict, # Vehicle pose record for the timestamp of the LIDAR
        }
    'radar': [{            # A list of Radar frame dictionaries (one for each Radar)
        token: str,        # Radar sensor record token
        filename: str,     # Pointcloud filename, relative to nuscenes root dir
        channel: str,      # Radar channel (always RADAR_BACK_LEFT)
        sc_record: dict,   # Radar sensor calibration parameters
        pose_record: dict, # Vehicle pose record for the timestamp of the Radar
        }, ...
    ],
    'anns': [{},...]       # All annotations records for this sample
    'sample_token': str,   # Nuscenes sample token
    'coordinates': str,    # Reference coordinate system ('vehicle', 'global')
    'meta': dict,          # Frame meta-data
    'id': int              # Frame ID
}
```

Annotation frame:
```yaml
ann = {
    'category_id': str,                       # Category token
    'num_lidar_pts': str,                     # Number of lidar points in this box
    'num_radar_pts': str,                     # Number of radar points in this box
    'instance_token': str,                    # Which object instance is this annotating. An instance can have multiple annotations over time.
    'distance': float,                        # Distance from the reference sensor (meters)
    'box_3d': nuscenes.utils.data_classes.Box # Instance of the nuscenes box describing this object
}
```
