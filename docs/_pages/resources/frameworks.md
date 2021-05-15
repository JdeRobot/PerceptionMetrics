---
layout: home
title: Frameworks
permalink: /resources/frameworks/


sidebar:
  nav: "resources"
---

Detection Metrics gives support for a set of different deep learning frameworks: Darknet, Tensorflow, Keras, PyTorch and Caffe.
Here, information on how to use each one of them with Detection Metrics is provided.

## Darknet
To use Darknet as your framework you only need OpenCV installed, which is a prerequisite.


## TensorFlow
First of all, you need tensorflow installed in your system, and you can get it installed by running the following commands.

* ### Installation (Skip if Tensorflow is already installed)

  ```
    pip install numpy==1.14.2 six==1.11.0 protobuf==3.5.2.post1
  ```
  For GPU use:
  ```
    pip install tensorflow_gpu
  ```
  For CPU only use:
  ```
    pip install tensorflow
  ```

For using TensorFlow as your framework, you would need a TensorFlow Trained Network. Some sample networks/models are available at [TensorFlow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

Download one of them, uncompress it and place it into the weights directory.

We will be using a COCO trained model for this example, but you can choose any model. Although you would have to create a class names file for that particular dataset written in a correct order.

Sample ```coco.names``` file for COCO dataset: [coco.names](https://github.com/JdeRobot/DetectionStudio/blob/master/samples/names/coco.names).
All it contains is a list of classes being used for this dataset in the correct order.
Place this file in the names directory.

Now create an empty ```foo.cfg``` file and place it in the cfg directory. It is empty because TensorFlow doesn't require any cfg file, just the frozen inference graph.

All done! Now you are ready to go!

Below, an example video using SSD MobileNet COCO on TensorFlow framework in Detection Metrics.

{% include video id="AWVdt7djJBg" provider="youtube" %}

## Keras

* ### Installation (skip if Keras is already installed)

    ```
    pip install numpy==1.14.2 six==1.11.0 protobuf==3.5.2.post1 h5py==2.7.1
    pip install Keras
    ```

For using Keras you must first have a Keras Trained Model Weights which are typically stored in an HDF5 file.
**No configuration** file is needed since new versions of Keras now contain architecture, weights and optimizer state all in a single HDF5 file. [See docs for the same](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).
  
Some sample pre-trained models are available at our [model zoo](../model_zoo), on different architectures and datasets.

## Caffe

For using Caffe, you will require OpenCV with it's dnn module built.
Steps for installing OpenCV 4.2 are available in [installation](../../installation)

To Use Caffe you would require some pre-trained models on Caffe, some are available at our own [model zoo](../model_zoo). But wait, you will also need to add custom parameters for Caffe, and our model zoo contains those parameters for each of the inferencer, just directly use that. 

## PyTorch

Install Pytorch using `pip install torch`. 
A `.yml` file is needed as configuration. The structure of the configuration file should contain the following:

```
modelPath: /path/to/model
modelName: model_name **CLASS NAME**
modelParameters: 100,True **PARAMETER THE CLASS NEEDS, IN ORDEN AND SEPARATED WITH COMMAS**
importName: model_import **IMPORT NAME FOR THE MODEL import model**
```