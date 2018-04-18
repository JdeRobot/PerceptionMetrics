# DeepLearningSuite
DeepLearning Suite is a set of tool that simplify the evaluation of most common object detection datasets with several object detection neural networks.

The idea is to offer a generic infrastructure to evaluates object detection algorithms againts a dataset and compute most common statistics:
* Intersecion Over Union
* Precision
* Recall



##### Supported datasets formats:
* YOLO
* Jderobot recorder logs
* Princeton RGB dataset [1]
* Spinello dataset [2]

##### Supported object detection frameworks/algorithms
* YOLO (darknet)
* TensorFlow
* Background substraction



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

### JDEROBOT
#### Deps

```
    sudo apt-get install -y libboost-filesystem-dev libboost-system-dev libboost-thread-dev libeigen3-dev libgoogle-glog-dev \
          libgsl-dev libgtkgl2.0-dev libgtkmm-2.4-dev libglademm-2.4-dev libgnomecanvas2-dev libgoocanvasmm-2.0-dev libgnomecanvasmm-2.6-dev \
          libgtkglextmm-x11-1.2-dev libyaml-cpp-dev icestorm zeroc-ice libxml++2.6-dev qt5-default libqt5svg5-dev libtinyxml-dev \
          catkin libssl-dev
```

#### Jderobot ThirdParty libraries:
```
    git clone https://github.com/JdeRobot/ThirdParty
    cd ThirdParty
    cd qflightinstruments
    qmake  qfi.pro
    make -j4
    make install

```

#### Jderobot

```bash
    git clone https://github.com/JdeRobot/JdeRobot
    cd JdeRobot
    cmake . -DENABLE_ROS=OFF
    make -j4
    cmake .
    sudo make install
```

### Darknet (jderobot fork)
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

``` make -j4 ``` <br>
``` sudo make -j4 install ```



Change <DARKNET_DIR> to your custom installation path.

# How to compile DL_DetectionSuite:
Once you have all the deps installed just:
```
    git clone https://github.com/JdeRobot/DeepLearningSuite
    cd DeepLearningSuite
    cd DeepLearningSuite/
    cmake . -DDARKNET_PATH=<DARKNET_DIR>
```


# Testing detectionsuite
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

In order to use darknet to detect objectd over the images you have to download the network configuration and the network weights [5] and [6]. Then set the corresponding paths into DeepLearningSuite/appConfig.txt. You have also to create a file with the corresponding name for each class detection for darknet, you can download the file directly from [7]

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
