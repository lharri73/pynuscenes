.. _autodoc:

API
===

Dataset
-------

.. autoclass:: pynuscenes.nuscenes_dataset.NuscenesDataset
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __len__, __getitem__

Utils
-----

.. autofunction:: pynuscenes.utils.nuscenes_utils.map_pointcloud_to_camera

.. autofunction:: pynuscenes.utils.nuscenes_utils.map_pointcloud_to_image

.. autofunction:: pynuscenes.utils.nuscenes_utils.map_annotation_to_camera

.. autofunction:: pynuscenes.utils.nuscenes_utils.box_3d_to_2d_simple

.. autofunction:: pynuscenes.utils.nuscenes_utils.boxes_in_image

.. autofunction:: pynuscenes.utils.nuscenes_utils.split_scenes

.. autofunction:: pynuscenes.utils.nuscenes_utils.global_to_vehicle

.. autofunction:: pynuscenes.utils.nuscenes_utils.vehicle_to_global

.. autofunction:: pynuscenes.utils.nuscenes_utils.vehicle_to_sensor

.. autofunction:: pynuscenes.utils.nuscenes_utils.sensor_to_vehicle

.. autofunction:: pynuscenes.utils.nuscenes_utils.get_box_dist

.. autofunction:: pynuscenes.utils.nuscenes_utils.bbox_to_corners

.. autofunction:: pynuscenes.utils.nuscenes_utils.corners3d_to_image

.. autofunction:: pynuscenes.utils.nuscenes_utils.quaternion_to_ry
