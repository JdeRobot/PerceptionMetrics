FROM ubuntu:16.04


#cuda9

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y aptitude
RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update
COPY ./keyboard /etc/default/keyboard
RUN apt-get install -y cuda

#common depds
RUN   apt-get install -y build-essential git openssh-client cmake libopencv-dev rapidjson-dev libboost-dev sudo


#jderobot
RUN apt-get install -y libboost-filesystem-dev libboost-system-dev libboost-thread-dev libeigen3-dev libgoogle-glog-dev \
    libgsl-dev libgtkgl2.0-dev libgtkmm-2.4-dev libglademm-2.4-dev libgnomecanvas2-dev libgoocanvasmm-2.0-dev libgnomecanvasmm-2.6-dev \
    libgtkglextmm-x11-1.2-dev libyaml-cpp-dev icestorm zeroc-ice libxml++2.6-dev qt5-default libqt5svg5-dev libtinyxml-dev \
    catkin libssl-dev


RUN useradd -ms /bin/bash docker
RUN 	echo "docker ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER docker
WORKDIR /home/docker


USER root
RUN apt-get install -y libboost-program-options-dev
USER docker

ENV DARKNET_DIR=/home/docker/install/darknet
ENV LIBRARY_PATH=/usr/local/cuda/lib64/:usr/local/cuda/targets/x86_64-linux/lib/
ENV LD_LIBRAR_PATH=/usr/local/cuda/lib64/:usr/local/cuda/targets/x86_64-linux/lib/


RUN   mkdir -p devel && cd devel && mkdir install && \
        git clone https://github.com/JdeRobot/darknet && \
        cd darknet && \
        cmake . -DCMAKE_INSTALL_PREFIX=$DARKNET_DIR && \
        make -j4 && \
        make -j4 install



RUN   cd devel && \
        git clone https://github.com/JdeRobot/DeepLearningSuite && \
        cd DeepLearningSuite && mkdir build && \
        cd build/ && \
        cmake . -DDARKNET_PATH=$DARKNET_DIR  && \
	make -j4
