---
layout: splash
title: Installation
permalink: /installation/

---

<br>
# Install DetectionSuite application
 
The application can be directly downloaded from the [repository releases](https://github.com/JdeRobot/DetectionSuite/releases/tag/continuous).

To run the app, first give executable permissions running:  
```
    chmod a+x DetectionSuitexxxxx.AppImage
```
and run it using:  
```
    ./DetectionSuitexxxxx -c configFile
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

## Using Docker  
First you need to have docker installed in your computer. If no, assuming you are in Ubuntu, install it using the following command:

```
    sudo apt install docker.io
```

Now you need to start the docker deamon using
```
    sudo service docker start
```

After starting the deamon , get DetectionSuite docker file using
```
    sudo docker pull jderobot/dl-detectionsuite
```

Now run the docker image using
```
    sudo docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix jderobot/dl-detectionsuite
```
where the ```-it``` flag instructs Docker to allocate a pseudo-TTY connected to the container’s stdin; creating an interactive ```bash``` shell in the container ,```-e``` tag is used to set the environment variable, in this case ```DISPLAY```, this is required because you will be using GUI for interaction through out rest of the tutorials.

Voila!! You can now use DetectionSuite. Proceed to [beginner's tutorial](../resources/tutorial/) to get started.

# Compile and Install from source
To use the latest version of DetectionSuite you need to compile and install it from source.
To get started you can either read along or follow [these video tutorials](https://www.youtube.com/watch?v=HYuFFTnEn5s&list=PLgB5c9xg9C91DJ30WFlHfHAhMECeho-gU).
## Requirements

### Common deps


<table>
<tr>
<td>
<center><b>Ubuntu</b></center>
</td>
<td>
<center><b>MacOS</b></center>
</td>
</tr>
<tr>
<td>
<pre>
sudo apt install build-essential git cmake rapidjson-dev libssl-dev
sudo apt install libboost-dev libboost-filesystem-dev libboost-system-dev libboost-program-options-dev
</pre>
</td>
<td>
<pre>
sudo easy_install numpy
brew install cmake boost rapidjson
</pre>
</td>
</tr>
<tr>
<td>
<pre>
sudo apt install libgoogle-glog-dev libyaml-cpp-dev qt5-default libqt5svg5-dev
</pre>
</td>
<td>
<pre>
brew install glog yaml-cpp qt
</pre>
<br/>
Also, just add qt in your PATH by running:<br/>
<pre>
echo 'export PATH="/usr/local/opt/qt/bin:$PATH"' >> ~/.bash_profile
</pre>
</td>
</tr>
<tr>
<td>
Install OpenCV 3.4
<pre>
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 3.4
mkdir build && cd build
cmake -D WITH_QT=ON -D WITH_GTK=OFF ..
make -j4
sudo make install
</pre>
</td>
<td>
<pre>
brew install opencv
</pre>

</td>
</tr>
</table>

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
DetectionSuite can currently read ROS and ICE Camera Streams. So, to enable Streaming support, install any one of them.

* ### Inferencing Frameworks
DetectionSuite currently supports many Inferencing Frameworks namely Darknet, TensorFlow, Keras and Caffe.
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
    For using Caffe as an inferencing framework, it is necessary to install OpenCV 3.4 or greater.


**Note:** Be Sure to checkout [functionality](../functionality/command_line_application) for tutorials on how to use the above mentioned functionalities and frameworks.  

# How to compile DetectionSuite:

Once you have all the required dependencies installed just run:

```
    git clone https://github.com/JdeRobot/DetectionSuite
    cd DetectionSuite/DeepLearningSuite
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

## Starting with DetectionSuite
The best way to start is with our [beginner's tutorial](../resources/tutorial/) for DetectionSuite.

If you have any issue feel free to drop a mail <vinay04sharma@icloud.com> or create an [issue](https://github.com/JdeRobot/DetectionSuite/issues) for the same.
