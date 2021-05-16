<a href="https://mmg-ai.com/en/"><img src="https://jderobot.github.io/assets/images/logo.png" width="100 " align="right" /></a>
# Detection Metrics

[![Publish Docker image](https://github.com/JdeRobot/DetectionStudio/actions/workflows/main.yml/badge.svg)](https://github.com/JdeRobot/DetectionStudio/actions/workflows/main.yml)

#### More info and documentation [here](https://jderobot.github.io/DetectionMetrics/).

Detection Metrics is a set of tools to evaluate object detection neural networks models over the common object detection datasets.
The tools can be accessed using the GUI or the command line applications. In the picture below, the general architecture is displayed.

![general_architecture](docs/assets/images/architecture.png)

The tools provided are:
* [Viewer](https://jderobot.github.io/DetectionMetrics/functionality/viewer/): view the dataset images with the annotations.
* [Detector](https://jderobot.github.io/DetectionMetrics/functionality/detector/): run a model over a dataset and get generate a new annotated dataset.
* [Evaluator](https://jderobot.github.io/DetectionMetrics/functionality/evaluator/): evaluate the ground truth dataset with another one and get the comparison metrics.
* [Deployer](https://jderobot.github.io/DetectionMetrics/functionality/deployer/): run a model over different inputs like a video or webcam and generate a new annotated dataset.
* [Converter](https://jderobot.github.io/DetectionMetrics/functionality/converter/): convert a dataset into another dataset format.
* [Command line application (CLI)](https://jderobot.github.io/DetectionMetrics/functionality/command_line_application/): access Detection Metrics toolset through command line
* [Detection Metrics as ROS Node](https://jderobot.github.io/DetectionMetrics/functionality/ros_node/): use Detection Metrics as a ROS Node.
+ [Labelling](https://jderobot.github.io/DetectionMetrics/resources/gsoc_19/): add or modify labels in the datasets in runtime when running Deployer.

The idea is to offer a generic infrastructure to evaluate object detection models against a dataset and compute the common statistics:
* mAP
* mAR
* Mean inference time.

# What's supported in Detection Metrics.

| Support | Detail                                                  |
| ------ | ------------------------------------------------------------ |
| Supported OS  | Multiplatform using Docker                 |
| Supported datasets  | <ul><li>COCO</li><li>ImageNet</li><li>Pascal VOC</li><li>Jderobot recorder logs</li><li>Princeton RGB dataset</li><li>Spinello dataset</li><li>Open images dataset</li></ul> |                |
| Supported frameworks   | <ul><li>TensorFlow</li><li>Keras</li><li>PyTorch</li><li>Yolo-OpenCV</li><li>Caffe</li><li>Background substraction</li></ul>  |
| Supported inputs in Deployer   | <ul><li>WebCamera/USB Camera</li><li>Videos</li><li>Streams from ROS</li><li>Streams from ICE</li><li>JdeRobot Recorder Logs</li></ul> |



# Installation

### Install packaged image

To quickly get started with Detection Metrics, we provide a docker image.

* Download docker image and run it
```
    docker run -dit --name detection-metrics -v [local_directory]:/root/volume/ -e DISPLAY=host.docker.internal:0 jderobot/detection-metrics:latest
```

This will start the GUI, provide a configuration file (appConfig.yml can be used) and you are ready to go. Check out the [web](https://jderobot.github.io/DetectionMetrics) for more information

### Installation from source (Linux only)

Check the installation guide [here](https://jderobot.github.io/DetectionMetrics/installation/). This is also the recommended installation 
for **contributors**.

# Starting with Detection Metrics
Check out the [beginner's tutorial](https://jderobot.github.io/DetectionMetrics/resources/tutorial/).

# General Detection Metrics GUI

The top toolbar shows the different tools available.

<p align="center">
  <img  style="border: 5px solid black;" width="800" height="500" src="docs/assets/images/main_window.png" />
</p>


# Example of detection and console output in Detection Metrics

Two image views are displayed, one with the ground truth and the other with the detected annotations.
In the console output, log info is shown.

![detector](docs/assets/images/detector.png)




