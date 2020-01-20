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
    git clone https://github.com/mrnabati/cocoapi_plus.git
    cd cocoapi_plus
    pip install -e .
    ```

### Build nuscenes_dataset
After having the above dependencies, run:
```bash
git clone https://github.com/mrnabati/nuscenes_dataset.git
cd nuscenes_dataset
pip install -e .
```

## Getting Started

#### Frame Structure
Database frame:
```yaml
frame = {
    'camera': [{            # A list of camera frame dictionaries (One for each camera)
        token: str,         # Camera sensor record token
        filename: str,      # Image filename, relative to nuscenes root dir
        channel: str,       # Camera channel (e.g. CAM_FRONT_RIGHT)
        width: int,         # Image width
        height: int,        # Image height
        image_id: int,      # Image ID
        }, ...
    ],
    'lidar': {              # A LIDAR frame dictionaries
        token: str,         # LIDAR sensor record token
        filename: str,      # Pointcloud filename, relative to nuscenes root dir
        channel: str,       # LIDAR channel (always LIDAR_TOP)
        }
    'radar': [{             # A list of Radar frame dictionaries (one for each Radar)
        token: str,         # Radar sensor record token
        filename: str,      # Pointcloud filename, relative to nuscenes root dir
        channel: str,       # Radar channel (always RADAR_BACK_LEFT)
        }, ...
    ],
    'anns': [{},...]        # All annotations records for this sample
    'sample_token': str,    # Nuscenes sample token
    'coordinates': str,      # Reference coordinate system ('vehicle', 'global')
    'meta': dict,           # Frame meta-data
    'id': int               # Frame ID
}
```

Dataloader frames have the same format as database frame, with the addition of 
sensor data:

```yaml
frame = {
    'camera': [{
        image: nparray,      # Image from this camera
        sc_record: dict,     # Camera sensor calibration parameters
        pose_record: dict,   # Vehicle pose record for the timestamp of the camera
        image_id: int,
        token: str,
        filename: str,
        channel: str,
        width: int,
        height: int,
        }, ...
    ],
    'lidar': {
        pointcloud: ndarray, # LIDAR Pointcloud
        pose_record: dict,   # Vehicle pose record for the timestamp of the lidar
        token: str,
        filename: str,
        channel: str,
        },
    'radar': {
        pointcloud: ndarray, # Radar Pointcloud
        pose_record: dict,   # Vehicle pose record for the timestamp of Radar
        },
    'anns': [{},...]        # Filtered annotations as Box objects
    'ref_pose_record': {},  # Reference pose record used for mapping anns from global to vehicle
    'sample_token': str,
    'coordinates': str,
    'meta': dict,
    'id': int
}
```