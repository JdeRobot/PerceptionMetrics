---
layout: splash
permalink: /
header:
  overlay_color: "#5e616c"
  overlay_image: /assets/images/cover/test_header_shear_3.png
  #actions:
  #  - label: "<i class='fas fa-download'></i> Install now"
  #    url: "/installation/"
excerpt: 
  Evaluation of object detection dataset simplified
---


# What is Detection Studio?

Detection Studio is an application that provides a toolbox of utilities oriented to simplify the development and testing of solutions based on object detection.
The application comes with a GUI (based on Qt) but it can also be used through command line.


{% include video id="gDP9nWCL0Vg" provider="youtube" %}

# Toolbox

The application is designed to experiment with datasets and neural networks using various frameworks. Currently it comes with the following utilities:

* [Viewer](functionality/viewer/): view the dataset images with the annotations.
* [Detector](functionality/detector/): run a model over a dataset and get generate a new annotated dataset.
* [Evaluator](functionality/evaluator/): evaluate the ground truth dataset with another one and get the comparison metrics.
* [Deployer](functionality/deployer/): run a model over different inputs like a video or webcam and generate a new annotated dataset.
* [Converter](functionality/converter/): convert a dataset into another dataset format.
* [Command line application (CLI)](functionality/command_line_application/): access Detection Studio toolset through command line
* [Detection Studio as ROS Node](functionality/ros_node/): use Detection Studio as a ROS Node.
+ [Labelling](resources/gsoc_19/): add or modify labels in the datasets in runtime when running Deployer

Every tool in Detection Studio requires a config file to run, where the main parameters needed are provided. Currently, YAML config file format is supported. See below on how to create a custom config file.
Each tool may have different requirements for keys in config file, and they can be known by passing the ```--help``` flag when using Detection Studio from
the command line.

### Creating a custom ```appConfig.yml```

It is recommended to create and assign a dedicated directory for storing all datasets, weights and config files, for easier access and a cleaner ```appConfig.yml``` file.

For instance, we will be using ```/opt/datasets/``` for demonstration purposes.

Create the following directories in ```/opt/datasets/```: ```cfg```, ```names```, ```weights``` and ```eval```.

Again, these names are just examples and can be changed, but must also be changed in ```appConfig.yml```.

* ```cfg```: This directory will store config files for various networks. For example, [*yolo-voc.cfg*](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-voc.cfg).
* ```names```: This directory will contain class names for various datasets. For example, [*voc.names*](https://github.com/pjreddie/darknet/blob/master/data/voc.names).
* ```weights```: This directory will contain weights for various networks, such as [*yolo-voc.weights*](https://pjreddie.com/media/files/yolo-voc.weights) for YOLO or a frozen inference graph for Tensorflow trained networks.
* ```eval```: Evaluations path.

Once completed, you can create your own custom appConfig.yml like the one mentioned. For example:

```

datasetPath: /opt/datasets/

evaluationsPath: /opt/datasets/eval

weightsPath: /opt/datasets/weights

netCfgPath: /opt/datasets/cfg

namesPath: /opt/datasets/names

inferencesPath: /opt/datasets

```

Place your weights in weights directory, config files in cfg directory, classname files in names. And you are ready to go ⚡️ .

# General Detection Studio GUI

The top toolbar shows the different tools available.

{:refdef: style="text-align: center;"}
<img src="../assets/images/main_window.png" style="border: 5px solid black;" alt="GUI" title="GUI" width="1000" height="1000" />
{: refdef}



# Example of detection and console output in Detection Studio

Two image views are displayed, one with the ground truth and the other with the detected annotations.
In the console output, log info is shown.


{:refdef: style="text-align: center;"}
<img src="../assets/images/detector.png" style="border: 5px solid black;" alt="Detector" title="Detector" width="1000" height="1000" />
{: refdef}


