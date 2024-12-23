---
layout: home
title: Detector
permalink: /v1/functionality/detector/


sidebar:
  nav: "main_v1"
---

**Detector** runs over and input dataset containing images and outputs the detected objects (detection dataset) providing the network weights

The dataset created by **Detector** can be further used by *Evaluator* to be compared with ***ground truth boxes*** and generate ***evaluation metrics***.

Just like **Deployer**, **Detector** also needs network weight files, inferencer implementation, network configuration files and inferencer class names as input.

Furthermore, it also requires a dataset as input, requiring annotation files, dataset implementation and class names to perform detections on.

Also, an output folder is required to store the output detected dataset, which can be used in **Evaluator** for generating accuracy metrics.

### Command line use example

An example of config file would be:

```
    inputPath: /opt/datasets/weights/annotations/instances_val2017.json
    outputPath: /opt/datasets/output/new-dataset/
    readerImplementation: COCO
    inferencerImplementation: tensorflow
    inferencerConfig: /opt/datasets/cfg/foo.cfg
    inferencerWeights: /opt/datasets/weights/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb
    inferencerNames: /opt/datasets/names/coco.names
    readerNames: /opt/datasets/names/coco.names

```

With the config file, change the directory to ``Tools/Detector`` inside build and run

```
    ./detector -c appConfig.yml
```

This will output the new detections dataset to the folder described in the configuration.

### GUI use video example

{% include video id="DpK5gqwoSBc" provider="youtube" %}

Using the GUI, the use of the available tools can be easier for a user. In this case select the different options from
the lists and run it pressing **Detect**.

The tool will start making detections in the images dataset. A window with the images detections will pop up, as shown
in the example video.

The use of *Depth Images* (if available) is possible.