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


## What is DetectionSuite?

DetectionSuite consists of a set of utilities oriented to simplify developing and testing solutions based on object detection.

{% include video id="gDP9nWCL0Vg" provider="youtube" %}

# Utilities

## DeepLearningSuite

DeepLearningSuite is a tool designed to experiment upon datasets and networks using various frameworks. Currently it has the following utilities:

+ Auto Evaluator
+ Viewer
+ Converter
+ Detector
+ Evaluator
+ Deployer

Every tool in DeepLearningSuite requires a config file to run, and currently YAML file format is supported. See below on how to create a custom config file.
Each tool may have different requirements for keys in config file, and they can be known by passing the ```--help``` flag.

### Creating a custom ```appConfig.yml```
It is recommended to create and assign a dedicated directory for storing all datasets, weights and config files, for easier access and a cleaner ```appConfig.yml``` file.

For Instance we will be using ```/opt/datasets/``` for demonstration purposes.

Create the following directories in ```/opt/datasets/```: ```cfg```, ```names```, ```weights``` and ```eval```.

Again, these names are temporary and can be changed, but must also be changed in ```appConfig.yml```.

* ```cfg```: This directory will store config files for various networks. For example, [*yolo-voc.cfg*](https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.cfg).
* ```names```: This directory will contain class names for various datasets. For example, [*voc.names*](https://github.com/pjreddie/darknet/blob/master/data/voc.names).
* ```weights```: This directory will contain weights for various networks, such as [*yolo-voc.weights*](https://pjreddie.com/media/files/yolo-voc.weights) for yolo or a frozen inference graph for tensorflow trained networks.
* ```eval```: Evaluations path.

Once done, you can create your own custom appConfig.yml like the one mentioned.

```

datasetPath: /opt/datasets/

evaluationsPath: /opt/datasets/eval

weightsPath: /opt/datasets/weights

netCfgPath: /opt/datasets/cfg

namesPath: /opt/datasets/names

inferencesPath: /opt/datasets

```

Place your weights in weights directory, config files in cfg directory, classname files in names. And you are ready to go ⚡️ .

## Examples of input and output

![](../assets/images/screen1.png)  |  ![](../assets/images/screen2.png) 
![](../assets/images/screen3.png)  |  
