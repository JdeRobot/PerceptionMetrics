---
layout: home
title: Frameworks
permalink: /functionality/frameworks/


sidebar:
  nav: "docs"
---

## Darknet
<!-- As an example you can use Pascal VOC dataset on darknet format using the following instructions to convert to the desired format:
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

In order to use darknet to detect objects over the images, you have to download the network configuration and the network weights [5] and [6]. Then set the corresponding paths into DeepLearningSuite/appConfig.txt. You have also to create a file with the corresponding name for each class detection for darknet, you can download the file directly from [7]

Once you have your custom appConfig.txt( see [creating-a-custom-appconfigtxt]( #creating-a-custom-appconfigtxt) ) you can run the DatasetEvaluationApp.


[1] https://pjreddie.com/media/files/yolo-voc.weights <br>
[2] https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.cfg <br>
[3] https://github.com/pjreddie/darknet/blob/master/data/voc.names <br>

-->
To use Darknet as your framework it is necessary to install it from JdeRobot's Darknet Fork.

* ### Installation

  Darknet supports both GPU and CPU builds, and GPU build is enabled by default.
  If your computer doesn't have a NVIDIA Graphics card, then it is necessary to turn off GPU build in cmake by 
  passing ```-DUSE_GPU=OFF``` as an option in cmake.

   ```
    git clone https://github.com/JdeRobot/darknet
    cd darknet
    mkdir build && cd build
   ```

   For **GPU** users:<br>
   ```
    cmake -DCMAKE_INSTALL_PREFIX=<DARKNET_DIR> ..
   ```
   For **Non-GPU** users (CPU build):

   ```
    cmake -DCMAKE_INSTALL_PREFIX=<DARKNET_DIR> -DUSE_GPU=OFF ..
   ```
   Change ```<DARKNET_DIR>``` to your custom installation path.

    ```
    make -j4 
    ``` 
    ``` 
    sudo make -j4 install 
    ```

   **Note:** After installing Darknet using above methods, you have to pass Darknet installation directory as an option
 in DetectionSuite's CMake like `cmake -D DARKNET_PATH=<DARKNET_DIR> ..`. Now, this `<DARKNET_DIR>` 
   is the same as `<DARKNET_DIR>` passed above.
   Cmake will throw a warning if it couldn't find Darknet libraries, just look for that and you are all done :zap: :boom: .



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

For using TensorFlow as your framework, you would need a TensorFlow Trained Network. Some sample Networks/models are available at [TensorFlow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

Download one of them and untar it and place it into the weights directory.

We will be using a COCO trained model for this example, but you can choose any model. Although you would have to create a class names file for that particular dataset written in a correct order.

Sample coco.names file for COCO dataset: [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names).<br>
All it contains is a list of classes being used for this dataset in the correct order.
Place this file in the names directory.

Now create an empty foo.cfg file and place it in the cfg directory. It is empty because tensorflow doesn't require any cfg file, just the frozen inference graph.

All done! Now you are ready to go!

Sample video using SSD MobileNet COCO on TensorFlow framework in DetectionSuite.

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

For using Caffe, you will require OpenCV 3.4 or greater with it's dnn module built. Following are the steps to install OpenCV 3.4 from source, though we will be shipping compiled binary packages of the same in the future, so as to speed up installation.

* ### Installation
    ```
    git clone https://github.com/opencv/opencv.git
    git checkout 3.4
    cmake -D WITH_QT=ON -D WITH_GTK=OFF ..
    make -j4
    sudo make install
    ```
    
Then, just build again, and just look out for a warning stating ```OpenCV 3.4 not Found, Caffe Support will be disabled```. If that warning persists than OpenCV 3.4 or higher hasn't been installed correctly, any you may want to look into that, else you are Good to Go.

To Use Caffe you would require some pre-trained models on Caffe, some are available at our own [model zoo](../model_zoo). But wait, you will also need to add custom parameters for Caffe, and our model zoo contains those parameters for each of the inferencer, just directly use that. 
