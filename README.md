# DetectionStudio

[![Build Status](https://travis-ci.org/JdeRobot/DetectionSuite.svg?branch=master)](https://travis-ci.org/JdeRobot/DetectionSuite)

### Please refer to the official documentation webpage [here](https://jderobot.github.io/DetectionStudio/).

DetectionStudio is a set of tool that simplify the evaluation of most common object detection datasets with several object detection neural networks.

The idea is to offer a generic infrastructure to evaluates object detection algorithms againts a dataset and compute most common statistics:
* Intersecion Over Union
* Precision
* Recall

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
* Background substraction

##### Supported Inputs for Deploying Networks
* WebCamera/ USB Camera
* Videos
* Streams from ROS
* Streams from ICE
* JdeRobot Recorder Logs


### Sample generation Tool
Sample Generation Tool has been developed in order to simply the process of generation samples for datasets focused on object detection. The tools provides some features to reduce the time on labelling objects as rectangles.


# Installation

Check the intallation guide [here](https://jderobot.github.io/DetectionStudio/installation/).

# Starting with DetectionStudio
The best way to start is with our [beginner's tutorial](https://jderobot.github.io/DetectionStudio/functionality/tutorial/) for DetectionStudio.
If you have any issue feel free to drop a mail <vinay04sharma@icloud.com> or create an issue for the same.

### DetectionStudio's sample images

GUI             |  Screenshot 1
:-------------------------:|:-------------------------:
![](docs/assets/images/detection_studio_gui.png)  |  ![](./docs/assets/images/screen1.png)   
Screenshot 2             |  Screenshot 3
![](./docs/assets/images/screen2.png) |  ![](./docs/assets/images/screen3.png)

