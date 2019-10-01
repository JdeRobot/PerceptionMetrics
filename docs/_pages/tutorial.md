---
layout: home
title: Tutorial
permalink: /functionality/tutorial/


sidebar:
  nav: "docs"
---

# Beginner's Tutorial Part 1
Sample Video depicting results of this tutorial:  
<p align="center">
<kbd><a href="http://www.youtube.com/watch?feature=player_embedded&v=xX2c_Trp9qY" target="_blank"><img src="http://img.youtube.com/vi/xX2c_Trp9qY/0.jpg"
alt="IMAGE ALT TEXT HERE" width="480" height="360" border="8"/></a>
</kbd>
</p>



DetectionSuite Supports lot's of functionalities for both Datasets and Inferencers, some of them are Viewing different datasets(Viewer), Converting Datasets from one to another(Converter), Generating Detection Datasets(Detector), Evaluating Datasets using Inferencers (Evaluator, Auto Evaluator ) and finally Deploying Datasets.

DetectionSuite has both GUI Application and Command Line Tools to execute its functionalities.

But as a beginner on should always start with GUI app, namely ```DatasetEvaluationApp```. We will start with ```Deployer Tab``` in the same app.

Once you've build DeepLearningSuite ( for building, [refer here](https://github.com/JdeRobot/dl-DetectionSuite#how-to-compile-dl_detectionsuite) ), navigate to DatasetEvaluationApp directory in the build directory. There you will find an executable, to run that you need the following.

* Config file
* Inferencer

Just create some directories using the following commands:
```
sudo mkdir /opt/datasets            # Root directory containing all datasets and inferencers
sudo mkdir /opt/datasets/weights    # Directory containing Inferencer weights
sudo mkdir /opt/datasets/cfg         # Directory containing Config files for different inferencers
sudo mkdir /opt/datasets/names       # Contains txt files listing classnames for various datasets
sudo mkdir /opt/datasets/output      # Evaluations output
```
Also, give this directory read and write permissions, so that files can be read from and written to this directory.
```
sudo chmod -R 666 /opt/datasets
```

For this tutorial we will be needing Inferencer weights, config files and Class names file only, and can be downloaded from here.

* Inferencer Weights and Config Files: [Our Model Zoo](Model-Zoo). You can choose any model for various FrameWorks trained on different datasets.
**Note:** TensorFlow and Keras doesn't require any config files, just create a `foo.cfg` file in the ```cfg``` folder.

* Datasets Class Names File: Can be Downloded from [here](ClassNames) .




For this tutorial, we will be using SSD MobileNet v1, trained on COCO, and would require following files:
- Model: [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz).
- Config: Just Create an empty foo.cfg file in `cfg/`.
- ClassNames: Download [COCO ClassNames File](coco.names).



Now, we will create `YAML` config file, to be passed as a parameter in DatasetEvaluationApp.

Sample config file for above created directory structure.
```
datasetPath: /opt/datasets/

evaluationsPath: /opt/datasets/eval

weightsPath: /opt/datasets/weights

netCfgPath: /opt/datasets/cfg

namesPath: /opt/datasets/names
```

if you have created directories, as mentioned above then just copy paste, above data into ```appConfig.yml```, else make changes accordingly.  

Now, we are good to go :zap: :boom: !!

Just type ``` ./DatasetEvaluationApp -c appConfig.yml ```

And you will see a GUI pop like this:  
![DetectionSuite StartUp](DetectionSuiteStartUp.png "DetectionSuite StartUp")

Configuring Parameters:
* **Deployer Implementation:** Method to fetch input images, can be `Camera`, `Video` or `Streams`. For video, select a video file. For streams, select between ROS or ICE, and enter the config parameters manually or select a config YAML file.

* **Net Weights:** Select weights, for the inferencer to be used.

* **Net Config:** Select Config file for the corresponding inferencer.

* **Inferencer Implementation:** Can be tensorflow, keras, Caffe or Darknet.

* **Inferencer Names:** Dataset ClassNames using which the inferencer was trained.

* **Save Inferenced Samples (optional):** If you require to save inferenced images with bounding box detections or masks, classIds and Confidence Scores, then check it and select the output folder.

* **Inferencer Params (For Caffe Only):** Parameters specific to inferencer. Some sample parameters for the same are listed on our [Model Zoo](Model-Zoo).
