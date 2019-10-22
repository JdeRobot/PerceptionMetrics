---
layout: home
title: Automatic Evaluation
permalink: /functionality/automatic_evaluation/


sidebar:
  nav: "docs"
---

DetectionSuite supports both a Qt based user interface and some command line based applications both requiring a config file to run.
Some users might prefer using the command line tools which can give results in a single run without the need to use the Graphical User Interface.

One such significant tool is **Automatic Evaluator**, which can evaluate multiple networks on a single dataset or multiple datasets in a single run.

All you need is config file containing details about the dataset(s) and network(s).

The results are then written in CSV files in the output directory specified.

To run this tool simply build this repository and navigate to ```build/Tools/AutoEvaluator```
and run ```./autoEvaluator -c config.yml```

Here ```config.yml``` is your required config file and some examples to create them are detailed below.

### Creating Config File

Given below is a sample config file to run **Automatic Evaluator** on COCO dataset for 2 inferencers.

```
Datasets:

-
  inputPath: /opt/datasets/coco/annotations/instances_train2014.json
  readerImplementation: COCO
  readerNames: /opt/datasets/names/coco.names


Inferencers:

-
  inferencerWeights:         /opt/datasets/weights/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb
  inferencerConfig:          /opt/datasets/cfg/foo.cfg
  inferencerImplementation:  tensorflow
  inferencerNames:           /opt/datasets/names/coco.names

-
  inferencerWeights:         /opt/datasets/weights/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb
  inferencerConfig:          /opt/datasets/cfg/foo.cfg
  inferencerImplementation:  tensorflow
  inferencerNames:           /opt/datasets/names/coco.names


outputCSVPath: /opt/datasets/output
```


As you can see there are two networks being used for inferencing: ```SSD_MobileNet``` and ```SSD_Inception```. Therefore, Inferencers contain an array of size 2.


### Using Multiple Frameworks

Below is a sample file for inferencing using multiple frameworks:

```
Datasets:

-
  inputPath: /opt/datasets/coco/annotations/instances_train2014.json
  readerImplementation: COCO
  readerNames: /opt/datasets/names/coco.names


Inferencers:

-
  inferencerWeights:         /opt/datasets/weights/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb
  inferencerConfig:          /opt/datasets/cfg/foo.cfg              # TensorFlow doesn't need any config file, hence any 
  inferencerImplementation:  tensorflow                             # empty foo.cfg file
  inferencerNames:           /opt/datasets/names/coco.names

-
  inferencerWeights:         /opt/datasets/weights/VGG_VOC0712_SSD_512x512_iter_120000.h5
  inferencerConfig:          /opt/datasets/cfg/foo.cfg              # New version Keras also doesn't need any file, all the
  inferencerImplementation:  keras                                  # data is stored in the HDF5 file including model 
  inferencerNames:           /opt/datasets/names/voc.names          # weights, configuration and optimizer state, hence we 
                                                                    # are using an empty foo.cfg file
-
  inferencerWeights:         /opt/datasets/weights/VGG_VOC0712_SSD_512x512_iter_240000.h5
  inferencerConfig:          /opt/datasets/cfg/foo.cfg              
  inferencerImplementation:  keras                             
  inferencerNames:           /opt/datasets/names/voc.names



outputCSVPath: /opt/datasets/output
```



**Note:** In the above example, you can see that a VOC trained network is being used to evaluate on COCO Ground Truth. This tool supports such evaluation by mapping Pascal VOC class names to COCO class names.
This mapping is very robust, it can also map synonyms and subclasses.

