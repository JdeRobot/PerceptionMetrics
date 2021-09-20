---
layout: splash
title: Installation
permalink: /installation/

---
<br>

# Install Detection Metrics using Docker

To quickly get started with Detection Metrics, we provide a docker image.

* Download docker image and run it
```
    docker run -dit --name detection-metrics -v [local_directory]:/root/volume/ -e DISPLAY=host.docker.internal:0 jderobot/detection-metrics:noetic
```

This will start the GUI, provide a configuration file (appConfig.yml can be used) and you are ready to go. Check out [functionality](/functionality/detector) for more information



# Install Detection Metrics from source for developers (only Linux)
 
To use the latest version of Detection Metrics you need to compile and install it from source.

## Requirements

### Common deps


| Ubuntu   |      MacOS      |  
|:-------------:|:-------------:|
| `sudo apt install build-essential git cmake rapidjson-dev libssl-dev` <br> `sudo apt install libboost-dev libboost-filesystem-dev libboost-system-dev libboost-program-options-dev python-dev python-numpy` | `sudo easy_install numpy` <br> `brew install cmake boost rapidjson` | 
       

| Ubuntu   |      MacOS      |  
|:-------------:|:-------------:|
| `sudo apt install libgoogle-glog-dev libyaml-cpp-dev qt5-default libqt5svg5-dev` <br> `sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev` <br> `sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev` <br> `sudo apt-get install libxvidcore-dev libx264-dev` |    `brew install glog yaml-cpp qt` <br> Also, just add qt in your PATH by running: <br> `echo 'export PATH="/usr/local/opt/qt/bin:$PATH"' >> ~/.bash_profile`   |


### Inferencers

Install only the inferencers that you need.

* ### OpenCV 4.2 (with CUDA GPU support) including Darknet

If you don't need GPU support (only applicable for Darknet YOLO with OpenCV), just ignore cmake options related with CUDA and GPU.

| Ubuntu   |      MacOS      |  
|:-------------:|:-------------:|
| `cd ~ `<br> `wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip` <br> `wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip` <br> `unzip opencv.zip <br> unzip opencv_contrib.zip` <br> `mv opencv-4.2.0 opencv` <br> `mv opencv_contrib-4.2.0 opencv_contrib` | `brew install opencv` |
| `cd ~/opencv` <br> `mkdir build` <br> `cd build` | |
|  You must change *CUDA_ARCH_BIN* version to yours GPU architecture version. <br> `cmake -D WITH_QT=ON -D WITH_GTK=OFF -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D CUDA_ARCH_BIN=**7.0 [CHANGE THIS ONE]** -D WITH_CUBLAS=1 -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D BUILD_opencv_python2=ON -D WITH_FFMPEG=1 -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules ..` | |
| `make -j8` | |
| `sudo make install` | |

Reference: [How to use OpenCV DNN module with Nvdia GPUs, CUDA and CUDNN](https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/)

* ### Tensorflow 

To use Tensorflow and/or Keras, you would need to install them.

To install Tensorflow:
```
    pip install tensorflow
```
or

```
    pip install tensorflow-gpu
```

* ### Keras 

To install Keras:
```
    pip install keras
```

* ### PyTorch 


To install PyTorch:
```
    pip install torch
```


### Optional Dependencies

#### CUDA (For GPU support)

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
Detection Metrics can currently read ROS and ICE Camera Streams. So, to enable Streaming support, install any one of them.


**Note:** Be Sure to checkout [functionality](../functionality/command_line_application) for tutorials on how to use the above mentioned functionalities and frameworks.  

# How to compile Detection Metrics:

Once you have all the required dependencies installed just run:

```
    git clone https://github.com/JdeRobot/DetectionMetrics
    cd DetectionMetrics/DetectionMetrics
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

## Starting with Detection Metrics
The best way to start is with our [beginner's tutorial](../resources/tutorial/) for Detection Metrics.

If you have any issue feel free to drop a mail <vinay04sharma@icloud.com> or create an [issue](https://github.com/JdeRobot/DetectionMetrics/issues) for the same.
