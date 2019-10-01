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

In order to use darknet to detect objects over the images you have to download the network configuration and the network weights [5] and [6]. Then set the corresponding paths into DeepLearningSuite/appConfig.txt. You have also to create a file with the corresponding name for each class detection for darknet, you can download the file directly from [7]

Once you have your custom appConfig.txt you can run the DatasetEvaluationApp.

### Using TensorFlow:
TensorFlow can also be used for Object Detection in this tool. All you need is a frozen inference graph and a video to run inferences on.
Some sample pre-trained graphs are available at [TensorFlow Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) trained on datasets such as COCO, KITTI and Open Images.

More instructions and tutorials on using the same are mentioned [here(Github's Wiki)](https://github.com/JdeRobot/dl-DetectionSuite/wiki).


# References.
[1] http://tracking.cs.princeton.edu/dataset.html \
[2] http://www2.informatik.uni-freiburg.de/~spinello/RGBD-dataset.html \
[3] YOLO: https://pjreddie.com/darknet/yolo/ \
[4] YOLO with c++ API: https://github.com/jderobot/darknet \
[5] https://pjreddie.com/media/files/yolo-voc.weights \
[6] https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.cfg \
[7] https://github.com/pjreddie/darknet/blob/master/data/voc.names
