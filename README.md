# DetectionStudio

[![Build Status](https://travis-ci.org/JdeRobot/DetectionSuite.svg?branch=master)](https://travis-ci.org/JdeRobot/DetectionSuite)

#### More info and documentation [here](https://jderobot.github.io/DetectionStudio/).

Detection Studio is a set of tool to evaluate object detection neural networks models over the common object detection datasets.
The tools can be accessed using the GUI or the command line applications. In the picture below, the general architecture is displayed.

![general_architecture](docs/assets/images/architecture.png)

The tools provided are:
* Viewer
* Detection
* Evaluator
* Deployer
* Converter
* Command line application (CLI)
* Detection Studio as ROS Node

The idea is to offer a generic infrastructure to evaluate object detection models against a dataset and compute the common statistics:
* mAP
* mAR
* Mean inference time.

#### Supported Operating Systems:
* Linux
* MacOS


##### Supported datasets formats:
* YOLO
* COCO
* ImageNet
* Pascal VOC
* Jderobot recorder logs
* Princeton RGB dataset [1]
* Spinello dataset [2]

##### Supported object detection frameworks/algorithms
* YOLO (darknet)
* TensorFlow
* Keras
* Caffe
* PyTorch
* Background substraction

##### Supported Inputs for Deploying Networks
* WebCamera/ USB Camera
* Videos
* Streams from ROS
* Streams from ICE
* JdeRobot Recorder Logs


| Number | Description                                                  |
| ------ | ------------------------------------------------------------ |
| Tools   | Viewer, Detection, Evaluator, Deployer, Converter, Command line application (CLI), Detection Studio as ROS Node                               | WIP    |
| Supported OS  | Linux, MacOS                 |
| Supported datasets  | YOLO, COCO, ImageNet, Pascal VOC, Jderobot recorder logs, Princeton RGB dataset, Spinello dataset |                |
| Supported frameworks   | TensorFlow, Keras, PyTorch, Yolo-OpenCV, Caffe, Background substraction  |
| Supported inputs in Deployer   | WebCamera/USB Camera, Videos, Streams from ROS, Streams from ICE, JdeRobot Recorder Logs |



# Installation

Check the installation guide [here](https://jderobot.github.io/DetectionStudio/installation/).

# Starting with DetectionStudio
The best way to start is with the [beginner's tutorial](https://jderobot.github.io/DetectionStudio/functionality/tutorial/) for DetectionStudio.

