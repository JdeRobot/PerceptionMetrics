---
layout: home
title: Detector
permalink: /functionality/detector/


sidebar:
  nav: "docs"
---

**Detector** more or less works like **Deployer**, the only difference is that it is run on a dataset and creates a detection dataset, whereas **Deployer** is run on a video or live stream.

The dataset created by **Detector** is further used by evaluator to be compared with ***ground truth boxes*** and generate ***evaluation metrics***.

Just like **Deployer**, **Detector** also needs network weight files, inferencer implementation, network configuration files and inferencer class names as input.

Furthermore, it also requires a dataset as input, requiring annotation files, dataset implementation and class names to perform detections on.

Also, an output folder is required to store the output detected dataset, which can be further used in **Evaluator** for generating accuracy metrics.
