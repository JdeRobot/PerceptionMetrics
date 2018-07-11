# DeepLearningSuite
DeepLearning Suite is a set of tool that simplify the evaluation of most common object detection datasets with several object detection neural networks.

The idea is to offer a generic infrastructure to evaluates object detection algorithms againts a dataset and compute most common statistics:
* Intersecion Over Union
* Precision
* Recall



##### Supported datasets formats:
* YOLO
* COCO
* ImageNet
* Pascal VOC
* Jderobot recorder logs
* Princeton RGB dataset [1]
* Spinello dataset [2]

##### Supported object detection frameworks/algorithms
* YOLO (darknet)
* TensorFlow
* Keras
* Caffe
* Background substraction

##### Supported Inputs for Deploying Networks
* WebCamera/ USB Camera
* Videos
* Streams from ROS
* Streams from ICE
* JdeRobot Recorder Logs

# Sample generation Tool
Sample Generation Tool has been developed in order to simply the process of generation samples for datasets focused on object detection. The tools provides some features to reduce the time on labeling objects as rectangles.


# Requirements

### CUDA

```
   NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \

     NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    sudo apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
     echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \

     sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list' && \
     sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list'
```

Update and Install

```
sudo apt-get update
sudo apt-get install -y cuda
```

### Common deps
```
 sudo apt-get install -y build-essential git cmake rapidjson-dev libboost-dev sudo
```

### Opencv
```
sudo apt-get install libopencv-dev

```

### Dependencies (Currently being refined and reduced)

```
    sudo apt-get install -y libboost-filesystem-dev libboost-system-dev libboost-thread-dev libeigen3-dev libgoogle-glog-dev \
          libgsl-dev libgtkgl2.0-dev libgtkmm-2.4-dev libglademm-2.4-dev libgnomecanvas2-dev libgoocanvasmm-2.0-dev libgnomecanvasmm-2.6-dev \
          libgtkglextmm-x11-1.2-dev libyaml-cpp-dev icestorm zeroc-ice libxml++2.6-dev qt5-default libqt5svg5-dev libtinyxml-dev \
          catkin libssl-dev
```

## Optional Dependencies
Below is a list of Optional Dependencies you may require depending on your Usage.

* ### Camera Streaming Support
Detectionsuite can currently read ROS and ICE Camera Streams. So, to enable Streaming support install any one of them.

* ### Inferencing FrameWorks
DetectionSuite currently supports many Inferencing FrameWorks namely Darknet, TensorFlow, Keras and Caffe.
Each one of them has some Dependencies, and are mentioned below.

   Choose your Favourite one and go ahead.

   * #### Darknet (jderobot fork)
      Darknet supports both GPU and CPU builds, and GPU build is enabled by default.
      If your Computer doesn't have a NVIDIA Graphics card, then it is necessary to turn of GPU build in cmake by passing ```-DUSE_GPU=OFF``` as an option in cmake.

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

   ``` make -j4 ``` <br>
   ``` sudo make -j4 install ```

   * #### TensorFlow
   Only depedency for using TensorFlow as an Inferencing framework is TensorFlow.
   So, just install TensorFlow. Though it should be 1.4.1 or greater.

   * #### Keras
   Similarly, only dependency for using  Keras as an Inferencing is Keras only.

   * #### Caffe
   For using Caffe as an inferencing framework, it is necessary to install OpenCV 3.4 or greater.

**Note: ** Be Sure to checkout the Wiki Pages for tutorials on how to use the above mentioned functionalities and frameworks.  

# How to compile DL_DetectionSuite:

Once you have all the required Dependencies installed just:

```
    git clone https://github.com/JdeRobot/DeepLearningSuite
    cd DeepLearningSuite/
    mkdir build && cd build
    cmake ..
    make -j4

```
**NOTE:** To enable Darknet support just pass an optinal parameter in cmake `-D DARKNET_PATH ` equal to Darknet installation directory, and is same as `<DARKNET_DIR>` passed above in darknet installation.

Once it is build you will find various executables in different folders ready to be executed :smile:.

## Starting with DetectionSuite
The best way to start is with our [beginner's tutorial](https://github.com/JdeRobot/dl-DetectionSuite/wiki/Beginner's-Tutorial-to-DetectionSuite) for DetectionSuite.
If you have any issue feel free to drop a mail <vinay04sharma@icloud.com> or create an issue for the same.
