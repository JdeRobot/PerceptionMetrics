---
layout: home
title: Viewer
permalink: /functionality/viewer/


sidebar:
  nav: "docs"
---

**Viewer** tab is used to view various datasets. It reads the images and the annotation files to label them with their respective class names and displays the same.

Currently, it supports various datasets, like COCO, Imagenet, Pascal VOC, Princeton, Spinello, etc.
It also supports displaying and labelling depth images by converting them into a human readable depth map.

Below is an example of using **Viewer** to view COCO Dataset.
To begin with, one would require COCO Dataset and the same can be downloaded from this [link](http://cocodataset.org/#download).

Download both the annotations and Train Val images, and put them in the same folder and then extract.

Now, change your ```appConfig.txt``` to include this folder's path (containing both annotations and images) or it's parent's path in dataset path.

Now, you can run DetectionSuite, switch to viewer tab, select the annotation file for COCO, which will be ```instances_trainxxxx.json```.
Select reader Implementation as COCO and class name as coco.names (can be downloaded from [here](https://github.com/pjreddie/darknet/blob/master/data/coco.names)).

And just click View!


Sample Video Demonstrating the same:
{% include video id="VMd6ve8brTE" provider="youtube" %}
