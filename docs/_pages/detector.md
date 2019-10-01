---
layout: home
title: Detector
permalink: /functionality/detector/


sidebar:
  nav: "docs"
---

## Detector
Detector more or less works like Deployer, the only difference is that it is run on a dataset and creates a detection Dataset, whereas deployer is run on a video or live stream.

The Dataset created by Detector is further used by evaluator to be compared with Ground Truth Boxes and generate Evaluation Metrics.

Just like Deployer, Detector also needs Network Weight Files, Inferencer Implementation, Network Configuration Files and Inferencer Class Names as Input.

FurtherMore, it also requires a Dataset as Input, requiring Annotation Files, Dataset Implementation and Class Names to perform Detections on.

Also, an output Folder is required to store the output Detected Dataset, which can be further used in Evaluator for generating accuracy metrics.
