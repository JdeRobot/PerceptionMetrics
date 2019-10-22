---
layout: home
title: Model zoo
permalink: /functionality/model_zoo/


sidebar:
  nav: "docs"
---

## TensorFlow Models
TensorFlow models can be downloaded from [TensorFlow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), or from our own set of models available [here](http://wiki.jderobot.org/store/deeplearning-networks/TensorFlow/).

## Keras Models
We have made our own set of Keras Models, compatible with the latest version, requiring on the HDF5 file to load the model completely.
Here's the [link](http://wiki.jderobot.org/store/deeplearning-networks/Keras/).

## Caffe Models
Caffe Models can be downloaded from [here](https://github.com/opencv/opencv/tree/master/samples/dnn#model-zoo) and the config files are available [here](https://github.com/opencv/opencv_extra/tree/master/testdata/dnn). We will soon index them at our own server, for easier access. 


|    Model | Scale |   Size WxH|   Mean subtraction | Channels order |
|---------------|-------|-----------|--------------------|-------|
| [MobileNet-SSD, Caffe](https://github.com/chuanqi305/MobileNet-SSD/) | 0.00784 (2/255) | 300x300 | 127.5 127.5 127.5 | BGR |
| [OpenCV face detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) | 1.0 | 300x300 | 104 177 123 | BGR |
| [SSDs from TensorFlow](https://github.com/tensorflow/models/tree/master/research/object_detection/) | 0.00784 (2/255) | 300x300 | 127.5 127.5 127.5 | RGB |
| [YOLO](https://pjreddie.com/darknet/yolo/) | 0.00392 (1/255) | 416x416 | 0 0 0 | RGB |
| [VGG16-SSD](https://github.com/weiliu89/caffe/tree/ssd) | 1.0 | 300x300 | 104 117 123 | BGR |
| [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn) | 1.0 | 800x600 | 102.9801 115.9465 122.7717 | BGR |
| [R-FCN](https://github.com/YuwenXiong/py-R-FCN) | 1.0 | 800x600 | 102.9801 115.9465 122.7717 | BGR |
| [Faster-RCNN, ResNet backbone](https://github.com/tensorflow/models/tree/master/research/object_detection/) | 1.0 | 300x300 | 103.939 116.779 123.68 | RGB |
| [Faster-RCNN, InceptionV2 backbone](https://github.com/tensorflow/models/tree/master/research/object_detection/) | 0.00784 (2/255) | 300x300 | 127.5 127.5 127.5 | RGB |
