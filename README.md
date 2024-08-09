# leaf_grasp_ROS
ROS1 Node for finding the optimal leaf to grasp from depth and semantic data.


## 1. Installation
Tested on Ubuntu 20.04, ROS Noetic

Install Conda env
```bash
conda env create -f conda_environment.yaml
```
Activate Conda env
```bash
conda activate leaf_processing
```

## 2. Running the Node

Start the Node:

```
rosrun leaf_grasp_ROS leaf_grasp.py
```

## Node I/O

# Subscribes to:
  `/depth_image` -> depth message \
  `/leaves_masks` -> masks message

# Publishes to:
  `/point_loc` -> grasp message

Besides subscribing to the topics above, the node also expects a point cloud generated by the depth map saved somewhere on the computer. The location the node expects to find the point cloud can be found on line 108 of the main node. Prior to publishing the depth map, I would recommend generating and saving the point cloud from the depth map using the intrinsics of whatever stereo camera configuration you are working with.
  
