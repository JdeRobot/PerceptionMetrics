---
layout: home
title: Deployer
permalink: /functionality/deployer/


sidebar:
  nav: "docs"
---

It can be used to run inferences on images using various frameworks (TensorFlow, Darknet, Keras and Caffe are currently supported). Below are instructions for using Deployer:

***Note:*** To access **Deployer** tool, one would need to run **Dataset Evaluation** app, which is a GUI for evaluating deep learning models.
To Run it navigate to the ``` DatasetEvaluationApp/build```. 
And simply type 
```
    ./DatasetEvaluationApp -c appConfig.yml 
```

## Input Images
In Order to run inferences, one needs input images and they can be captured from multiple resources mentioned below:
* Video
* Camera (WebCam, USBCam, etc. directly connected to the system)
* Stream: 
    * ROS
    * ICE
* JdeRobot Recorder Logs


### Video
To use video just select video form ***Deployer Input Imp*** list and select the corresponding video file for the same in the **Deployer** tab.

### Camera
To run camera simply select camera from ***Deployer Input Imp*** list and you are good to go.
It will automatically select a camera.

### Stream
Currently, DetectionSuite supports ROS (Robot Operating System) and ICE (Internet Communication Engine) for reading streams, and both of them are optional dependencies and required only if you plan to use one of them.
After selecting stream from the ***Deployer Input Imp*** list you can choose between the following:

   * ### ROS
        To use ROS, just select ROS from Inferencer Implementation Params, and enter the corresponding params for 
        the same.
        Also, if you have a yaml file containing params, you can select it instead.

   * ### ICE
        Similarly, for ICE just select IT and enter the corresponding params or a yaml file as you please.

**Note:** If you find any one of the above disabled then, it isn't probably installed or you DetectionSuite didn't find it.

## Network
As any other tool, you would need a network to infer, on any one of the supported frameworks. Just fill up the following parameters, so as to let DetectionSuite know more about the Inferencer.
    
   * ### 1. Net Weights:
        Select the Network's Weights file. It would be .pb (frozen inference graph) for TensorFlow, 
        .h5 for Keras, .caffemodel for Caffe and .weights for Darknet.
   * ### 2. Net Configuration File:
        Configuration files aren't necessary for TensorFlow and Keras (any empty file would suffice), but for Darknet would need ```.cfg``` and for Caffe a ```.prototxt``` file.
   * ### 3. Inferencer Names (or Network Class Names):
        These are the class names on which the deep learning network was trained on. So, a file containing a list of class names in the correct order.
        See [datasets](../class_names/) for some class names file.
   * ### 4. Inferencer Implementation (Framework being Used):
        The framework using which the Deep Learning Network was trained and we currently support Darknet, TensorFlow, Keras and Caffe.  

**Note:** For Caffe, you might need to add some additional parameters specific to your model. Some samples are available at our [model zoo](../model_zoo/).
   
After configuring all these parameters, you are good to go :zap: :boom: .  
  
Also, all these parameters at once might seem scary and tedious to configure, but after you launch the GUI it will seem quite easy and almost all of them are self-explanatory :relaxed: .