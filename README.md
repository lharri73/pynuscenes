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
```
frame = {
    'camera': [{
        token: str,
        filename: str,
        channel: str,
        }, ...
    ],
    'lidar': {
        token: str,
        filename: str,
        pc: nparray,
        channel: str,
    }
    'radar': [{
        token: str,
        filename: str,
        pc: nparray,
        channel: str,
        }, ...
    ],
    'sweeps': dict,
    'meta': dict,
    'id': sample_id
}

Dataloader frame:
```
frame = {
    'camera': [{
        token: str,
        filename: str,
        image: nparray,
        sc_record: dict,
        channel: str,
        }, ...
    ],
    'lidar': [{
        token: str,
        filename: str,
        pc: nparray,
        channel: str,
        }, ...
    ],
    'radar': [{
        token: str,
        filename: str,
        pc: nparray,
        channel: str,
        }, ...
    ],
    'sweeps': dict,
    'meta': dict,
    'id': sample_id
}
```