---
layout: home
title: Testing DetectionSuite
permalink: /functionality/testing_detectionsuite/


sidebar:
  nav: "docs"
---

As an example you can use Pascal VOC dataset on darknet format using the following instructions to convert to the desired format:
```bash
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar

wget https://pjreddie.com/media/files/voc_label.py
python voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
```

In order to use Darknet to detect objects over the images you have to download the [network configuration](https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.cfg) and the [network weights](https://pjreddie.com/media/files/yolo-voc.weights). Then set the corresponding paths into DeepLearningSuite/appConfig.txt. 
You have also to create a file with the corresponding name for each class detection for Darknet, you can download the file directly from [here](https://github.com/pjreddie/darknet/blob/master/data/voc.names).

Once you have your custom appConfig.txt you can run the DatasetEvaluationApp.

### Using TensorFlow:
TensorFlow can also be used for Object Detection in this tool. All you need is a frozen inference graph and a video to run inferences on.
Some sample pre-trained graphs are available at [TensorFlow Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) trained on datasets such as COCO, KITTI and Open Images.

More instructions and tutorials on using the same are mentioned [here](../tutorial/).