# ROS package for official YOLOv7

This repo contains a ROS noetic package for the official YOLOv7. It wraps the 
[official implementation](https://github.com/WongKinYiu/yolov7) into a ROS node (so most credit 
goes to the YOLOv7 creators).

Also credit goes to lucazso for starting thsi repo.

### Note
There are currently two YOLOv7 variants out there. This repo contains the 
implementation from the paper [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696).

## Requirements & Getting Started

Following ROS packages are required:
- [vision_msgs](http://wiki.ros.org/vision_msgs)
- [geometry_msgs](http://wiki.ros.org/geometry_msgs)
- [shape_msgs](http://wiki.ros.org/shape_msgs)
- [message_generation](http://wiki.ros.org/message_generation)
- [actionlib_msgs](http://wiki.ros.org/actionlib_msgs)
Yo can install them with:
```
sudo apt-get install ros-noetic-vision-msgs
sudo apt-get install ros-noetic-geometry-msgs
sudo apt-get install ros-noetic-shape-msgs
sudo apt-get install ros-noetic-message-generation
sudo apt-get install ros-noetic-actionlib-msgs
```
First, clone the repo into your catkin workspace and build the package:
```
git clone https://github.com/robertokcanale/yolov7-ros.git ~/catkin_ws/src/
cd ~/catkin_ws
catkin build yolov7_ros
```

The Python requirements are listed in the `requirements.txt`. You can simply 
install them as
```
pip install -r requirements.txt
```

Download the YOLOv7 weights from the [official repository](https://github.com/WongKinYiu/yolov7).

The package has been tested under Ubuntu 20.04 and Python 3.8.10.

## Usage
## COCO Object Detection
Before you launch the node, adjust the parameters in the 
[launch file](launch/yolov7.launch). For example, you need to set the path to your 
YOLOv7 weights and the image topic to which this node should listen to. The launch 
file also contains a description for each parameter.

```
roslaunch yolov7_ros yolov7.launch
```

## YOLOv7 Human Pose Estimation
Before you launch the node, adjust the parameters in the 
[launch file](launch/yolov7_hpe.launch). For example, you need to set the path to your 
YOLOv7 weights and the image topic to which this node should listen to. The launch 
file also contains a description for each parameter.
You can download the weights from the official repo or here:
https://drive.google.com/file/d/1Khl44NDNp2bpQMWWN-hvfc258SGx_QtV/view?usp=sharing

```
rosrlaunch yolov7_ros yolov7_hpe.launch
```

Each time a new image is received it is then fed into YOLOv7.

### Notes
- The detections will be published under `/yolov7/out_topic`.
- If you set the `visualize` parameter to `true`, the detections will be drawn into 
  the image, which is then published under `/yolov7/out_topic/visualization`.

## Coming Soon
- ROS2 implementation
