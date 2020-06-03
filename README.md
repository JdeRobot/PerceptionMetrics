# DetectionStudio

[![Build Status](https://travis-ci.org/JdeRobot/DetectionSuite.svg?branch=master)](https://travis-ci.org/JdeRobot/DetectionSuite)

#### More info and documentation [here](https://jderobot.github.io/DetectionStudio/).

Detection Studio is a set of tools to evaluate object detection neural networks models over the common object detection datasets.
The tools can be accessed using the GUI or the command line applications. In the picture below, the general architecture is displayed.

![general_architecture](docs/assets/images/architecture.png)

The tools provided are:
* [Viewer](https://jderobot.github.io/DetectionStudio/functionality/viewer/): view the dataset images with the annotations.
* [Detector](https://jderobot.github.io/DetectionStudio/functionality/detector/): run a model over a dataset and get generate a new annotated dataset.
* [Evaluator](https://jderobot.github.io/DetectionStudio/functionality/evaluator/): evaluate the ground truth dataset with another one and get the comparison metrics.
* [Deployer](https://jderobot.github.io/DetectionStudio/functionality/deployer/): run a model over different inputs like a video or webcam and generate a new annotated dataset.
* [Converter](https://jderobot.github.io/DetectionStudio/functionality/converter/): convert a dataset into another dataset format.
* [Command line application (CLI)](https://jderobot.github.io/DetectionStudio/functionality/command_line_application/): access Detection Studio toolset through command line
* [Detection Studio as ROS Node](https://jderobot.github.io/DetectionStudio/functionality/ros_node/): use Detection Studio as a ROS Node.

The idea is to offer a generic infrastructure to evaluate object detection models against a dataset and compute the common statistics:
* mAP
* mAR
* Mean inference time.

# What's supported in Detection Studio.

| Support | Detail                                                  |
| ------ | ------------------------------------------------------------ |
| Supported OS  | Linux, MacOS                 |
| Supported datasets  | YOLO, COCO, ImageNet, Pascal VOC, Jderobot recorder logs, Princeton RGB dataset, Spinello dataset |                |
| Supported frameworks   | TensorFlow, Keras, PyTorch, Yolo-OpenCV, Caffe, Background substraction  |
| Supported inputs in Deployer   | WebCamera/USB Camera, Videos, Streams from ROS, Streams from ICE, JdeRobot Recorder Logs |



# Installation

Check the installation guide [here](https://jderobot.github.io/DetectionStudio/installation/).

# Starting with DetectionStudio
The best way to start is with the [beginner's tutorial](https://jderobot.github.io/DetectionStudio/functionality/tutorial/) for DetectionStudio.

