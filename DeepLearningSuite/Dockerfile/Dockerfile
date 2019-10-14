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
RUN   apt-get install -y build-essential git openssh-client cmake rapidjson-dev libboost-dev python-dev python-numpy sudo


#jderobot
RUN apt-get install -y libboost-filesystem-dev libboost-system-dev libboost-program-options-dev libeigen3-dev libgoogle-glog-dev \
    libgsl-dev libyaml-cpp-dev qt5-default libqt5svg5-dev libtinyxml-dev libssl-dev


RUN useradd -ms /bin/bash docker
RUN 	echo "docker ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER docker
WORKDIR /home/docker

ENV DARKNET_DIR=/home/docker/install/darknet
ENV LIBRARY_PATH=/usr/local/cuda/lib64/:usr/local/cuda/targets/x86_64-linux/lib/
ENV LD_LIBRAR_PATH=/usr/local/cuda/lib64/:usr/local/cuda/targets/x86_64-linux/lib/

RUN  git clone https://github.com/opencv/opencv.git && \
	cd opencv && git checkout 3.4 && \
  mkdir build && cd build && \
	cmake -D WITH_QT=ON -D WITH_GTK=OFF .. && \
	make -j4 && \
	sudo make install && cd /home/docker

RUN   mkdir -p devel && cd devel && mkdir install && \
        git clone https://github.com/JdeRobot/darknet && \
        cd darknet && \
        cmake . -DCMAKE_INSTALL_PREFIX=$DARKNET_DIR && \
        make -j4 && \
        make -j4 install



RUN   cd devel && \
        git clone https://github.com/JdeRobot/dl-DetectionSuite && \
        cd dl-DetectionSuite/DeepLearningSuite && mkdir build && \
        cd build/ && cmake -DDARKNET_PATH=$DARKNET_DIR ..  && \
        make -j4
