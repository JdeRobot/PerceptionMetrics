---
layout: splash
title: Installation
permalink: /installation/

---

<br>
# Install Detection Studio application
 
The application can be directly downloaded from the [repository releases](https://github.com/JdeRobot/DetectionStudio/releases/tag/continuous).

To run the app, first give executable permissions running:  
```
    chmod a+x DetectionStudioxxxxx.AppImage
```
and run it using:  
```
    ./DetectionStudioxxxxx -c configFile
```


Requirements: `python` & `numpy`.

To use Tensorflow and/or Keras, you would need to install them.

To install Tensorflow:
```
    pip install tensorflow
```
or

```
    pip install tensorflow-gpu
```
To install Keras:
```
    pip install keras
```

To install Pytorch:
```
    pip install torch
```

# Compile and Install from source
To use the latest version of Detection Studio you need to compile and install it from source.
To get started you can either read along or follow [these video tutorials](https://www.youtube.com/watch?v=HYuFFTnEn5s&list=PLgB5c9xg9C91DJ30WFlHfHAhMECeho-gU).
## Requirements

### Common deps


| Ubuntu   |      MacOS      |  
|:-------------:|:-------------:|
| `sudo apt install build-essential git cmake rapidjson-dev libssl-dev` <br> `sudo apt install libboost-dev libboost-filesystem-dev libboost-system-dev libboost-program-options-dev` | `sudo easy_install numpy` <br> `brew install cmake boost rapidjson` | 
       

| Ubuntu   |      MacOS      |  
|:-------------:|:-------------:|
| `sudo apt install libgoogle-glog-dev libyaml-cpp-dev qt5-default libqt5svg5-dev` |    `brew install glog yaml-cpp qt` <br> Also, just add qt in your PATH by running: <br> `echo 'export PATH="/usr/local/opt/qt/bin:$PATH"' >> ~/.bash_profile`   |

### OpenCV 4.2 (with CUDA GPU support) 

If you don't need GPU support (only applicable for Darknet YOLO with OpenCV), just ignore cmake options related with CUDA and GPU.

| Ubuntu   |      MacOS      |  
|:-------------:|:-------------:|
| `cd ~ `<br> `wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip` <br> `wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip` <br> `unzip opencv.zip <br> unzip opencv_contrib.zip` <br> `mv opencv-4.2.0 opencv` <br> `mv opencv_contrib-4.2.0 opencv_contrib` | `brew install opencv` |
| `cd ~/opencv` <br> `mkdir build` <br> `cd build` | |
|  You must change *CUDA_ARCH_BIN* version to yours GPU architecture version. <br> `cmake -D WITH_QT=ON -D WITH_GTK=OFF -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D CUDA_ARCH_BIN=**7.0 [CHANGE THIS ONE]** -D WITH_CUBLAS=1 ..` | |
| `make -j8` | |
| `sudo make install` | |

Reference: [How to use OpenCV DNN module with Nvdia GPUs, CUDA and CUDNN](https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/)


## Optional Dependencies

### CUDA (For GPU support)

```
    NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \

    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
        sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
        sudo apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
        echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \

    sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list' && \
    sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list'
```

Update and install

```
    sudo apt-get update
    sudo apt-get install -y cuda
```

Below is a list of more optional dependencies you may require depending on your usage.

* ### Camera Streaming Support
Detection Studio can currently read ROS and ICE Camera Streams. So, to enable Streaming support, install any one of them.

* ### Inferencing Frameworks
Detection Studio currently supports many Inferencing Frameworks namely Darknet, TensorFlow, Keras, PyTorch and Caffe.
Each one of them has some dependencies, and are mentioned below.

    Choose your favourite one and go ahead.

    * #### Darknet (jderobot fork)
    Included in OpenCV libraries.

    * #### TensorFlow
    The only dependency for using TensorFlow as an inferencing framework is TensorFlow.
    So, just install TensorFlow. It should be 1.4.1 or greater.

    * #### Keras
    Similarly, the only dependency for using Keras as an inferencing framework is Keras.
    
    * #### Caffe
    To use Caffe as an inferencing framework, it is necessary to install OpenCV.


**Note:** Be Sure to checkout [functionality](../functionality/command_line_application) for tutorials on how to use the above mentioned functionalities and frameworks.  

# How to compile Detection Studio:

Once you have all the required dependencies installed just run:

```
    git clone https://github.com/JdeRobot/DetectionStudio
    cd DetectionStudio/DetectionStudio
    mkdir build && cd build
```
```
    cmake ..
```
**Note:** GPU support is enabled by default
```
    make -j4
```

Once it is built, you will find various executables in different folders ready to be executed :smile:.

## Starting with Detection Studio
The best way to start is with our [beginner's tutorial](../resources/tutorial/) for Detection Studio.

If you have any issue feel free to drop a mail <vinay04sharma@icloud.com> or create an [issue](https://github.com/JdeRobot/DetectionStudio/issues) for the same.
